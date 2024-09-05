""" Service constants for unit testing """
import os
import inspect
import configparser
import unittest
import urllib.parse
import requests
import requests.exceptions
import requests.utils
import unittest.test


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

tests = {'OPENAI':
    {
        'Model-ok'              : ['test_model','gpt-4o',200],
        'Model-bad'             : ['test_model','failing-llm',500],
        'Reload'                : ['test_reload','',200],
        'Clear'                 : ['test_clear','',200],
        'Cache'                 : ['test_cache',{
            'answer1' :"User:content='who wrote rag service?'",
            'answer2'           :"AI:content='RAG Service was developed by L1Blom.'"
        }, 200],
        'Temperature-ok'        : ['test_temperature',0.0,200],
        'Temperature-too-low'   : ['test_temperature',-0.6,500],
        'Temperature-too-high'  : ['test_temperature',2.1,500],
        'Prompt-text'           : ['test_prompt_text',{
            'prompt' : "prompt=who wrote rag service?",
            'answer' : "RAG Service was developed by L1Blom."
        }, 200],
        'Prompt-PDF'            : ['test_prompt_pdf',{
            'prompt' : "prompt=how many watchers has this GitHub library?",
            'answer' : "The GitHub repository for the RAG Service has 2 watchers."
        }, 200],
    }
}

class RagServiceMethods(unittest.TestCase):
    """ RAG tests """
    def test_model(self):
        """ Test model setting, correct or incorrect model according to LLM """
        for llm in tests.keys():
            for options in ['Model-ok','Model-bad']:
                option = tests[llm][options]
                with self.subTest(llm, option=option):
                    model       = option[1]
                    result_code = option[2]
                    try:
                        response = requests.get(api_url+"/model?model="+model, timeout=10000)
                        status = response.status_code
                    except requests.HTTPError as e:
                        print("Error! "+str(e))
                    self.assertEqual(status, result_code)

    def test_reload(self):
        """ Test reload of the data """
        try:
            response = requests.get(api_url+"/reload", timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)

    def test_clear(self):
        """ Test to clear the cache """
        try:
            response = requests.get(api_url+"/clear", timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)

    def test_chache(self):
        """ Test to print the contents of the cache """
        self.test_clear()
        self.test_prompt_text()
        answer1 = "User:content='who wrote rag service?'"
        answer2 = "AI:content='RAG Service was developed by L1Blom.'"
        try:
            response = requests.get(api_url+"/cache", timeout=10000)
            status = response.status_code
            result = response.content.decode("utf-8").split("\n")
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)
        self.assertEqual(result[0], answer1) # User:
        self.assertEqual(result[1], answer2) # AI:

    def test_temperature(self):
        """ Test to set temparature too high, low, within boundaries 0.0 and 2.0"""
        try:
            temp=str(2.1)
            response = requests.get(api_url+"/temp?temp="+temp, timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 500)

        try:
            temp=str(-0.1)
            response = requests.get(api_url+"/temp?temp="+temp, timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 500)

        try:
            temp=str(rc.get('DEFAULT','temperature'))
            response = requests.get(api_url+"/temp?temp="+temp, timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)

    def test_prompt_text(self):
        """ Test prompt """
        self.test_clear()
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            prompt = "prompt=who wrote rag service?"
            response = requests.post(api_url, data=prompt, headers=headers, timeout=10000)
            status = response.status_code
            result = response.content.decode("utf-8")
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)
        self.assertEqual(result,"RAG Service was developed by L1Blom.")

    def test_prompt_pdf(self):
        """ Test prompt """
        self.test_clear()
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            prompt = "prompt=how many watchers has this GitHub library?"
            response = requests.post(api_url, data=prompt, headers=headers, timeout=10000)
            status = response.status_code
            result = response.content.decode("utf-8")
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)
        self.assertEqual(result,"The GitHub library has 2 watchers.")

    def test_image(self):
        """ Test image """
        self.test_clear()
        try:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            data = {
                "prompt": "what is written?",
                "image" : api_url+"/file/Open-AI.jpg"
            }
            response = requests.post(api_url + '/image', data=data, headers=headers, timeout=10000)
            status = response.status_code
            result = response.content.decode("utf-8")
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)
        self.assertEqual(result[0:35], 'The text in the image reads "OpenAI."'[0:35])

if __name__ == '__main__':
    print(unittest.main())

