""" Service constants for unit testing """
import unittest
import urllib.parse
import requests
import requests.exceptions
import requests.utils
import constants.constants__unittest as constants


ID = constants.ID
api_url = urllib.parse.quote("http://localhost:"+
                             str(constants.PORT)+
                             "/prompt/"+ID, 
                             safe=':?/')

class RagServiceMethods(unittest.TestCase):
    """ RAG tests """
    def test_model(self):
        """ Test model setting, correct or incorrect model according to OpenAI """
        model = constants.MODELTEXT
        try:
            response = requests.get(api_url+"/model?model="+model, timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)
        # Test model setting, wrong model according to OpenAI
        model = "failing-llm"
        try:
            response = requests.get(api_url+"/model?model="+model, timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 500)

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

    def test_chache_content(self):
        """ Test to print the contents of the cache """
        self.test_clear()
        self.test_prompt()
        answer1 = "User:content='who wrote rag service?'"
        answer2 = "AI:content='The RAG Service was developed by L1Blom.'"
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
            temp=str(constants.TEMPERATURE)
            response = requests.get(api_url+"/temp?temp="+temp, timeout=10000)
            status = response.status_code
        except requests.HTTPError as e:
            print("Error! "+str(e))
        self.assertEqual(status, 200)

    def test_prompt(self):
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
        self.assertEqual(result,"The RAG Service was developed by L1Blom.")

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
