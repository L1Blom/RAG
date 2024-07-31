# RAG

RAG is a template service for a retrieval-augmented generator based on the examples of LangChain. See: [Build a Retrieval Augmented Generation (RAG) App](https://python.langchain.com/v0.2/docs/tutorials/rag/)

## Installation

1. Clone https://github.com/L1Blom/rag to your project directory
2. Choose an ID for your instance, like MyDocs
3. Create a constants/ dir under the project and copy constants_id.py to it as constants_MyDocs.py
4. Change the contents of this file to reflect your situation
5. Create a data/ dir and create a MyDocs/ directory in it
6. Add the files for your RAG context, like text-files
7. Create a Python virtual environment for your project (optional)
8. Modify the example services/rag.service_template file to point to the right place of your project directory
9. Copy the services file to /etc/systed/system/rag_MyDocs.service
10. Enable and start the service

```bash
sudo systemctl enable rag_MyDocs
sudo systemctl start rag_MyDocs
sudo systemctl status rag_MyDocs
```

## Commands

All calls support POST and GET.

1. /prompt/<ID>
    Parameter: prompt (string)
    Your prompt to be send to openAI

2. /prompt/<ID>/model
    Parameter: model (string)
    Your model to be used, like "gpt-4o"
    Currently no checking on valid models. Can result in http 500 error (non-fatal)

3. /prompt/<ID>/temp
    Parameter: temp (string, will be cast to float)
    Temperature setting, between 0.0 and 2.0
    Settings above 1.0 can give significant halicunations and degrades performance too.

4. /prompt/<ID>/reload
    Parameters: none
    After adding files to your data directory, use this to reload the vector store

5. /prompt/<ID>/clear
    Paramters: none
    Clears the cache, the in-memory history

6. /prompt/<ID>/cache
    Paramaters: none
    Prints the cache contents to the response object

7. /prompt/<ID>/image
    Parameters: prompt (string), image (URL to image)
    Uploads the image to openAI and use prompt to get the desired contents like: 'What is the mood of the persons?'
    Note: only works ik model is 'gpt-4o' 

## Usage

```bash
# change the model
curl -X POST --data-urlencode "model=gpt-4o" http://<your server>:<your port>/prompt/<ID>/model
Model set to: gpt-4o(your virtual env) 
# prompt to your data
curl -X POST --data-urlencode "prompt=your question?" http://<your server>:<your port>/prompt/<ID>
Your answer based on the context files provided in data/<ID>
```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)