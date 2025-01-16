# RAG Service

RAG Service is a template service for a retrieval-augmented generator based on the examples of LangChain. See: [Build a Retrieval Augmented Generation (RAG) App](https://python.langchain.com/v0.2/docs/tutorials/rag/)

This service can be used with curl but there is another project that serves the user interface:
[L1Blom/ragui](https://github.com/L1Blom/ragui)

## Technologies

[![Used in this project](https://skillicons.dev/icons?i=vscode,py,flask,ai,md,linux,bash,nginx,git,raspberrypi)](https://skillicons.dev)

## Installation

1. Clone [L1Blom/rag](https://github.com/L1Blom/rag) to your project directory
2. Move config.py_example to config.py and add your API Keys
3. Choose an ID for your instance, like MyDocs
4. Copy constants/constants.ini to constants_MyDocs.ini
5. Change the contents of this file to reflect your situation
6. Create a MyDocs/ directory in data/ and in MyDocs/ a directory vectorstore/ and html/
7. Add the files for your RAG context, like text and PDF files, to your MyDocs/ directory
8. Create a Python virtual environment for your project (optional)
9. Modify the example services/rag.service_template file to point to the right place of your project directory
10. Copy the services file to /etc/systed/system/rag_MyDocs.service
11. Enable and start the service

```bash
sudo systemctl enable rag_MyDocs
sudo systemctl start rag_MyDocs
sudo systemctl status rag_MyDocs
```

## Commands

All calls support POST and GET. For \<ID\> use your chosen ID like MyDocs

1. /prompt/\<ID\>/

    Parameter: prompt (string)

    Your prompt to be send

2. /prompt/\<ID\>/full

    Parameter: prompt (string)

    Your prompt to be send

    Returns all document fragments used for this prompt

3. /prompt/\<ID\>/search

    Parameter: prompt (string)

    Your prompt to be send

    Similar search in the local documents, returns fragments and scores

4. /prompt/\<ID\>/model

    Parameter: model (string)

    Your model to be used, like "gpt-4o"

    Checking on valid models with OpenAI client.models.list(). Can result in http 500 error (non-fatal)

5. /prompt/\<ID\>/temp

    Parameter: temp (string, will be cast to float)

    Temperature setting, between 0.0 and 2.0

    Settings above 1.0 can give significant halicunations and degrades performance too.

    Timeout can result in http 408 error (non-fatal)

6. /prompt/\<ID\>/reload

    Parameters: none

    After adding files to your data directory, use this to reload the vector store

7. /prompt/\<ID\>/clear

    Paramters: none

    Clears the cache, the in-memory history

8. /prompt/\<ID\>/cache

    Paramaters: none

    Prints the cache contents to the response object

9. /prompt/\<ID\>/modelnames

    Paramaters: none

    Prints the names of the possible models used in the selected APIs

10. /prompt/\<ID\>/params

    Paramaters: section (string), param (string)

    Prints the settings from the .ini file

11. /prompt/\<ID\>/image

    Parameters: prompt (string), image (URL to image)

    Uploads the image to openAI and use prompt to get the desired contents like: 'What is the mood of the persons?'

    Note: only works if model is set to 'gpt-4o'. Other models result in http 500 error (non-fatal)

12. /prompt/\<ID\>/upload

    Parameters: file (string) (maximum size 16 Mb)

    Uploads the file to the directory DATA_DIR, only if the extension is listed in DATA_GLOB_*
    If not, results in http 500 error (non-fatal)

## Usage

```bash
# change the model
curl -X POST --data-urlencode "model=gpt-4o" http://<your server>:<your port>/prompt/<ID>/model

Model set to: gpt-4o
# prompt to your data
curl -X POST --data-urlencode "prompt=your question?" http://<your server>:<your port>/prompt/<ID>

Your answer based on the context files provided in data/<ID>
```

## Constants file

important contstants are:

```python
# simple string like "myDocs"
ID = _unittest
# Directory that will be scanned for files to be added to the context
DATA_DIR=data/_unittest
# All the file extentions you want to be part of the context, see LangChain documentation
# Currently text and pdf are supported by RAG Service
DATA_GLOB_TXT = *.txt
DATA_GLOB_PDF = *.pdf
# Persistence directory for vectorstore
PERSISTENCE = data/_unittest/vectorstore
# Where the HTML files reside, also needed for the unit tests
HTML = data/_unittest/html
```

## Unit tests

To run the unit tests, run the program in the project directory using the ID '_unittest'.
It will start a local RAG service accessible at port 8888 (see constants__unittest.py for all defaults).
When it is running, unit tests can be performed.
Currently when USE_LLM is set to OPENAI, it will run smoothly.
Other settings like GROQ might fail depending on the licences you have because of too many calls per minute. if so, try to run the unit test one by one.
See below all possible API-calls and paramters:

```bash
<your virtual environment>/bin/python ragservice.py _unittest
INFO:root:Working directory is /home/leen/projects/rag
INFO:httpx:HTTP Request: GET https://api.openai.com/v1/models "HTTP/1.1 200 OK"
INFO:root:path -> /prompt/_unittest prompt
INFO:root:path -> /prompt/_unittest/full prompt
INFO:root:path -> /prompt/_unittest/search prompt,similar
INFO:root:path -> /prompt/_unittest/documents id
INFO:root:path -> /prompt/_unittest/params section,param
INFO:root:path -> /prompt/_unittest/globals 
INFO:root:path -> /prompt/_unittest/modelnames 
INFO:root:path -> /prompt/_unittest/embeddingnames 
INFO:root:path -> /prompt/_unittest/model model
INFO:root:path -> /prompt/_unittest/embeddings embedding
INFO:root:path -> /prompt/_unittest/chunk chunk_size,chunk_overlap
INFO:root:path -> /prompt/_unittest/temp temp
INFO:root:path -> /prompt/_unittest/reload 
INFO:root:path -> /prompt/_unittest/clear 
INFO:root:path -> /prompt/_unittest/cache 
INFO:root:path -> /prompt/_unittest/file file
INFO:root:path -> /prompt/_unittest/context file,action
INFO:root:path -> /prompt/_unittest/image image,prompt
INFO:root:path -> /prompt/_unittest/upload 
INFO:chromadb.telemetry.product.posthog:Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
INFO:root:Loaded 8 chunks from persistent vectorstore
INFO:root:Chain initialized: gpt-4o
 * Serving Flask app 'ragservice'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8888
 * Running on http://192.168.2.200:8888
INFO:werkzeug:Press CTRL+C to quit
```

Now you are able to run the unit tests:

```bash
<your virtual environament>/bin/python ragservice_unittest.py -v
Testing OPENAI
test_cache (__main__.RagServiceMethods.test_cache)
Test to print the contents of the cache ... User:content='who wrote rag service?' User:content='who wrote rag service?'
AI:content='RAG Service was developed by Leen Blom.' AI:content='RAG Service was developed by L1Blom.'
ok
test_clear (__main__.RagServiceMethods.test_clear)
Test to clear the cache ... ok
test_image (__main__.RagServiceMethods.test_image)
Test image ... ok
test_model (__main__.RagServiceMethods.test_model)
Test model setting, correct or incorrect model according to LLM ... ok
test_prompt (__main__.RagServiceMethods.test_prompt)
Test prompt ... ok
test_reload (__main__.RagServiceMethods.test_reload)
Test reload of the data ... ok
test_temperature (__main__.RagServiceMethods.test_temperature)
Test to set temparature too high, low, within boundaries 0.0 and 2.0 ... ok

----------------------------------------------------------------------
Ran 7 tests in 17.713s

OK
```

## TODO's and wishes

- None at the moment

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

The image used in the unittest is licensed [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) and was found at [Trusted Reviews](https://www.trustedreviews.com/versus/chat-gpt-4-vs-chat-gpt-3-4309130)
