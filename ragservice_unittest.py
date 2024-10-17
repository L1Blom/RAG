""" Service constants for unit testing """
import os
import re
import sys
import time
import inspect
import configparser
import unittest
import urllib.parse
import requests
import requests.exceptions
import requests.utils
import unittest.test
from config_unittest import tests

wd = os.path.abspath(inspect.getsourcefile(lambda:0)).split("/")
wd.pop()
os.chdir('/'.join(map(str,wd)))

base_dir = os.path.abspath(os.path.dirname(__file__)) + '/'

# Read the constants from a config file
rc = configparser.ConfigParser()
rc.read(base_dir + "constants/constants__unittest.ini")
ID   = rc.get('DEFAULT','id')
port = rc.get('FLASK','port')
api_url = urllib.parse.quote("http://localhost:"+
                             str(port)+
                             "/prompt/"+ID, 
                             safe=':?/')

def get_llm():
    try:
        response = requests.get(api_url+"/params?section=LLMS&param=USE_LLM", timeout=10000)
        status = response.status_code
        if status == 200:
            llm = response.content.decode("utf-8").split("\n")[0]
            if llm == "":
                print(f"Model USE_LLM not set in RAG paramters")
                sys.exit(1)
            if not llm in tests.keys():
                print(f"Model {llm} not found in config_unittest.py")
                sys.exit(1)
            print(f"Testing {llm}")
        else: 
            raise requests.HTTPError
    except requests.HTTPError as e:
        print("Error! Can't connect to RAG Service (forgotten to start?) "+str(e))       
    return llm

llm = get_llm()

def find_words(sentence,words,nowords):
    sentence_array = re.split(r'\W+', sentence)
    words_array = words = re.split(r'\W+', words)
    l = len(sentence_array)
    i = 0
    for word in words:
        if word in sentence_array:
            i = i+1
    if i >= nowords:
        return True
    else:
        return False
        
class RagServiceMethods(unittest.TestCase):
    """ RAG tests """
    
    def test_model(self):
        """ Test model setting, correct or incorrect model according to LLM """
        for options in ['Model-bad','Model-ok']:
            option = tests[llm][options]
            with self.subTest(llm, option=option):
                model           = option[1]
                expected_status = option[2]
                try:
                    response = requests.get(api_url+"/model?model="+model, timeout=10000)
                    status = response.status_code
                except requests.HTTPError as e:
                    print("Error! "+str(e))
                self.assertEqual(status, expected_status)

    def test_reload(self):
        """ Test reload of the data """
        option = tests[llm]['Reload']
        with self.subTest(llm, option=option):
            expected_status = option[2]
            try:
                response = requests.get(api_url+"/reload", timeout=10000)
                status = response.status_code
            except requests.HTTPError as e:
                print("Error! "+str(e))
            self.assertEqual(status, expected_status)

    def test_clear(self):
        """ Test to clear the cache """
        option = tests[llm]['Clear']
        with self.subTest(llm, option=option):
            expected_status = option[2]
            try:
                response = requests.get(api_url+"/clear", timeout=10000)
                status = response.status_code
            except requests.HTTPError as e:
                print("Error! "+str(e))
            self.assertEqual(status, expected_status)

    def test_cache(self):
        """ Test to print the contents of the cache """
        self.test_clear()
        self.test_prompt()
        option = tests[llm]['Cache']
        with self.subTest(llm, option=option):
            answer1 = option[1]['answer1']
            answer2 = option[1]['answer2']
            expected_status = option[2]
        try:
            response = requests.get(api_url+"/cache", timeout=10000)
            status = response.status_code
            result = response.content.decode("utf-8").split("\n")
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, expected_status)
        print(result[0],answer1)
        self.assertTrue(find_words(result[0],answer1,4))
        print(result[1],answer2)
        self.assertTrue(find_words(result[1],answer2,4))

    def test_temperature(self):
        """ Test to set temparature too high, low, within boundaries 0.0 and 2.0"""
        self.test_clear()
        for option in ['Temperature-too-high','Temperature-too-low','Temperature-ok']:
            option = tests[llm][option]
            with self.subTest(llm, option=option):
                temp   = str(option[1])
                expected_status = option[2]
                try:
                    response = requests.get(api_url+"/temp?temp="+temp, timeout=10000)
                    status = response.status_code
                except requests.HTTPError as e:
                    print("Error! "+str(e))
                self.assertEqual(status, expected_status)

    def test_prompt(self):
        """ Test prompt """
        self.test_clear()
        for option in ['Prompt-text','Prompt-PDF']:
            option = tests[llm][option]
            with self.subTest(llm, option=option):
                prompt = option[1]['prompt']
                answer = option[1]['answer']
                expected_status = option[2]
                try:
                    headers = {"Content-Type": "application/x-www-form-urlencoded"}
                    response = requests.post(api_url, data=prompt, headers=headers, timeout=10000)
                    status = response.status_code
                    result = response.content.decode("utf-8")
                except requests.HTTPError as e:
                    print("Error! "+str(e))
                self.assertEqual(status, expected_status)
                self.assertTrue(find_words(answer,result,3))

    @unittest.skipUnless(llm=="OPENAI","Runs with OPENAI only")
    def test_image(self):
        """ Test image """
        self.test_clear()
        option = tests[llm]['Image']
        with self.subTest(llm, option=option):
            prompt = option[1]['prompt']
            answer = option[1]['answer']
            expected_status = option[2]
            try:
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                data = {
                    "prompt": prompt,
                    "image" : api_url+"/file?file=Open-AI.jpg"
                }
                response = requests.post(api_url + '/image', data=data, headers=headers, timeout=10000)
                status = response.status_code
                result = response.content.decode("utf-8")
            except requests.HTTPError as e:
                print("Error! "+str(e))
            self.assertEqual(status, expected_status)
            if status == 200:
                self.assertTrue(find_words(answer,result,1))

    def tearDown(self):
        if llm == 'GROQ':
            time.sleep(10)  # sleep time in seconds

if __name__ == '__main__':
    unittest.main()
