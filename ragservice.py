"""_summary_
RAG for your own documents
"""
import logging
import os
import sys
import uuid
import base64
import inspect
import configparser
import subprocess
import pprint
import json
from dotenv import load_dotenv
from pathlib import PurePath
from typing import List
from urllib.request import urlopen
from urllib.parse import quote
import chromadb
from flask import Flask, make_response, request, send_file, Response
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import openai
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader,TextLoader, PyPDFDirectoryLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_groq import ChatGroq
from groq import Groq
import time

# The program needs a ID to be able to read the config
# Use _unittest for activating unittests or your own tests
if len(sys.argv) != 2:
    print("Error: argument missing -> ID")
    sys.exit(os.EX_USAGE)
rag_project=sys.argv[1]

# Read the secrets from the secrets directory
def read_secret(secret_path):
    with open(secret_path, 'r') as file:
        return file.read().strip()

# Load the environment variables from the env directory
if os.path.exists("env/config.env"):
    logging.info("Loading environment variables from env/config.env")
    load_dotenv("env/config.env")
# Load the secrets from the secrets directory
if os.path.exists("/run/secrets"):
    logging.info("Loading secrets from /run/secrets")
    os.environ['AZURE_OPENAI_APIKEY'] = read_secret('/run/secrets/azure_openai_apikey')
    os.environ['OPENAI_APIKEY'] = read_secret('/run/secrets/openai_apikey')
    os.environ['LLAMA3_APIKEY'] = read_secret('/run/secrets/llama3_apikey')
    os.environ['GROQ_APIKEY'] = read_secret('/run/secrets/groq_apikey')

# set working dir to program dir to allow relative paths in configs
wd = os.path.abspath(inspect.getsourcefile(lambda:0)).split("/")
wd.pop()
os.chdir('/'.join(map(str,wd)))

base_dir = os.path.abspath(os.path.dirname(__file__)) + '/'

# Read the constants from a config file
rc = configparser.ConfigParser()

try: 
    constantsfile = base_dir + "constants/constants_"+rag_project+".ini"
    rc.read(constantsfile)
    rcconfighost = rc.get('DEFAULT','config_server')
except Exception as e:
    print("No constants file found: " + constantsfile)
    sys.exit(os.EX_USAGE)

# max 16Mb for uploads
app = Flask(__name__)
maxmb = rc.getint('FLASK','max_mb_size')
app.config['MAX_CONTENT_LENGTH'] = maxmb * 1024 * 1024
CORS(app)

@app.context_processor
def context_processor():
    """ Store the globals in a Flask way """
    return dict()

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        res = Response()
        res.headers['X-Content-Type-Options'] = '*'
        return res



def get_modelnames(mode, modeltext, embedding_model=None):
    """ 
        Load all API keys to environment variables.
        Return possible model names and check the chosen one.
    """
    modelnames = []
    embeddingnames = []
    

    has_model_list = False
    match mode:
        case 'OPENAI':
            client = openai.OpenAI(api_key = os.environ.get('OPENAI_APIKEY')) 
            has_model_list = True
        case 'GROQ':
#            client = openai.OpenAI(api_key = os.environ.get('OPENAI_APIKEY')) 
            client = Groq(api_key = os.environ.get("GROQ_APIKEY"))
            has_model_list = True
        case 'OLLAMA':
            result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE).stdout.decode('utf-8')
            namelist = result.strip().split()
            for name in namelist:
                hit = name.find(':latest')
                if hit>0:
                    model = name[0:hit]
                    modelnames.append(model)
        case 'AZURE':
            modelnames = [rcmodel]
            embeddingnames = [embedding_model]

    if has_model_list:  # Must be made more explicit, Ollama doesn't have a client
        models = client.models.list().data
        embeddingnames = sorted([
            str(dict(modelitem)['id']) 
            for modelitem in models 
            if str(dict(modelitem)['id']).startswith(('text')) 
        ])
        modelnames = sorted([
            str(dict(modelitem)['id']) 
            for modelitem in models 
            if str(dict(modelitem)['id']).startswith(('gpt','o1','o3')) 
                and 'audio' not in str(dict(modelitem)['id'])
                and 'video' not in str(dict(modelitem)['id'])
        ])

    if not modeltext in modelnames:
        logging.error("Model %s not found in %s models", modeltext, mode)
        sys.exit(os.EX_CONFIG)

    if not embedding_model in embeddingnames:
        logging.error("Embedding %s not found in %s models", embedding_model, mode)
        sys.exit(os.EX_CONFIG)

    return modelnames, embeddingnames

# Configureer logging
logging.basicConfig(level=rc.get('DEFAULT','logging_level'))
logging.info("Working directory is %s", os.getcwd())

globvars        = context_processor()
globvars['Project']    = rag_project
globvars['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
rcllms          = globvars['USE_LLM']      = rc.get('LLMS','use_llm')
rcmodel         = globvars['ModelText']    = rc.get('LLMS.'+rcllms,'modeltext')
rcembedding     = globvars['Embedding']    = rc.get('LLMS.'+rcllms,'embedding_model')
rctemp          = globvars['Temperature']  = rc.getfloat('DEFAULT','temperature')
rcsimilar       = globvars['Similar']      = rc.getint('DEFAULT','similar') 
rcscore         = globvars['Score']        = rc.getfloat('DEFAULT','score') 
rcsysprompt1    = globvars['SystemPrompt'] = rc.get('DEFAULT','contextualize_q_system_prompt')
rcsysprompt2    = globvars['SystemPrompt'] = rc.get('DEFAULT','system_prompt')
rcchunksize     = globvars['ChunkSize']    = rc.getint('DEFAULT','chunk_size') 
rcchunkoverlap  = globvars['ChunkOverlap'] = rc.getint('DEFAULT','chunk_overlap') 

if rctemp < 0.0 or rctemp > 2.0:
    logging.error("Temperature not between 0.0 and 2.0: %f", rctemp)
    sys.exit(os.EX_CONFIG)
globvars['Chain']       = None
globvars['Store']       = {}
globvars['Session']     = uuid.uuid4()
globvars['VectorStore'] = None
globvars['NoChunks']    = 0

modelnames, embeddingnames = get_modelnames(rcllms, rcmodel, rcembedding)

def set_chat_model(temp=rctemp):
    match globvars['USE_LLM']:
        case "OPENAI":
            my_api_key=os.environ.get('OPENAI_APIKEY')
            globvars['LLM']     = ChatOpenAI(api_key=my_api_key,model=rcmodel,temperature=temp)
        case "OLLAMA":
            my_api_key=os.environ.get('OLLAMA_APIKEY')
            globvars['LLM']     = Ollama(model=rcmodel)
        case "GROQ":
            my_api_key=os.environ.get('GROQ_APIKEY')
            globvars['LLM']     = ChatGroq(api_key=my_api_key,model=rcmodel,temp=temp)
        case "AZURE":
            my_api_key=os.environ.get('AZURE_OPENAI_APIKEY')
            if my_api_key is None:
                logging.error("Azure API key not found")
                sys.exit(os.EX_CONFIG)  
            globvars['LLM']     = AzureAIChatCompletionsModel(
                endpoint=rc.get('LLMS.AZURE','azure_openai_model_endpoint'),
                credential=my_api_key,
                temperature=temp,
                model_name=rcmodel,
                verbose=True,
                client_kwargs={ "logging_enable": True }
            )

set_chat_model(rctemp)

if rcmodel not in modelnames:
    print(f"Error: modelname {rcmodel} not known with {rcllms}")
    sys.exit(os.EX_CONFIG)

@app.route('/ping', methods=['GET'])
@cross_origin()
def ping():
    """ Return the start time """
    pid = os.getpid()
    timestamp = globvars['timestamp']
    return {'answer':{'timestamp': timestamp, 'pid': pid}}
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """ Memory history store

    Args:
        session_id (str): no of the session

    Returns:
        BaseChatMessageHistory: the reference to the store
    """
    prefix = ""
    Store = globvars['Store']
    if session_id not in Store:
        Store[session_id] = InMemoryHistory()
    for message in Store[session_id].messages:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"
        logging.info("%s: %s",prefix, message.content)
    return Store[session_id]

def encode_image(image_url) -> base64:
    """ Encode image to base64 """
    logging.info("Encoding image: %s",image_url)
    with urlopen(image_url) as url:
        f = url.read()
        image = base64.b64encode(f).decode("utf-8")
    return image

def embedding_function() -> OpenAIEmbeddings:
    """ Return an Embedding"""
    match globvars['USE_LLM']:
        case "OPENAI":
            my_api_key=os.environ.get('OPENAI_APIKEY')
            return OpenAIEmbeddings(api_key=my_api_key,model=rcembedding)
        case "AZURE":
            my_api_key=os.environ.get('AZURE_OPENAI_APIKEY')
            if my_api_key is None:
                logging.error("Azure API key not found")
                sys.exit(os.EX_CONFIG)      
            return AzureAIEmbeddingsModel(
                endpoint=rc.get('LLMS.AZURE','azure_openai_embedding_endpoint'),
                credential=my_api_key,
                model_name=rcembedding
            ) 
        case _:
            return OpenAIEmbeddings()

def load_files(vectorstore, file_type):
    file_types = {'docx' : [UnstructuredWordDocumentLoader,'single'],
                  'pptx' : [UnstructuredPowerPointLoader,'by_title'],
                  'xlsx' : [UnstructuredExcelLoader,'single'],
                  'pdf'  : [PyPDFDirectoryLoader,'elements'],
                  'txt'  : [TextLoader,'elements']}  

    for ftype in file_types.keys():
        if file_type != 'all' and ftype != file_type:
                continue 
        loader_cls = file_types[ftype][0]
        mode = file_types[ftype][1]
        glob = rc.get('DEFAULT','data_glob_'+ftype)  

        logging.info("Loading %s files with glob %s",ftype,glob)
        splits = None
        match ftype:
            case 'txt':
                if rc.has_option('DEFAULT','LANGUAGE'):
                    text_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language[rc.get('DEFAULT','language')],
                        chunk_size=rc.getint('DEFAULT','chunk_size'),
                        chunk_overlap=rc.getint('DEFAULT','chunk_overlap'))
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=rc.getint('DEFAULT','chunk_size'),
                        chunk_overlap=rc.getint('DEFAULT','chunk_overlap'))
                text_loader_kwargs={'autodetect_encoding': True}

                loader = DirectoryLoader(path=rc.get('DEFAULT','data_dir'),
                            glob=glob,
                            loader_cls=loader_cls,
                            silent_errors=True,
                            loader_kwargs=text_loader_kwargs)
                docs = loader.load()
                splits = text_splitter.split_documents(docs)
                if len(splits) >0:
                    vectorstore.add_documents(splits)
                    logging.info("Context loaded from %s documents, %s splits",ftype, str(len(splits)))              
            case 'pdf':
                text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=rc.getint('DEFAULT','chunk_size'),
                            chunk_overlap=rc.getint('DEFAULT','chunk_overlap'))
                loader = PyPDFDirectoryLoader(path=rc.get('DEFAULT','data_dir'),
                            glob=glob)
                splits = loader.load_and_split()
                if len(splits) >0:
                    vectorstore.add_documents(splits)
                    logging.info("Context loaded from %s documents, %s splits",ftype, str(len(splits)))              
            case 'pptx':
                text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=rc.getint('DEFAULT','chunk_size'),
                        chunk_overlap=rc.getint('DEFAULT','chunk_overlap'))
                loader_kwargs={'autodetect_encoding': True,
                                'chunking_strategy': mode}
                loader = DirectoryLoader(path=rc.get('DEFAULT','data_dir'),
                            glob=glob,
                            loader_cls=loader_cls,
                            silent_errors=True,
                            loader_kwargs=loader_kwargs)
                docs = loader.load()
                for doc in docs:
                    if dict(doc)['metadata']:
                        mtd = dict(doc)['metadata']
                        for key in mtd:
                            if type(mtd[key]) == list:
                                mtd[key] = ','.join([str(item) for item in mtd[key]])
                        doc.metadata = mtd
                if len(docs) >0:
                    vectorstore.add_documents(docs)
                    logging.info("Context loaded from %s documents, %s docs",ftype, str(len(docs)))              
            case _:
                text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=rc.getint('DEFAULT','chunk_size'),
                        chunk_overlap=rc.getint('DEFAULT','chunk_overlap'))
                loader_kwargs={'autodetect_encoding': True, 'mode': mode}
                loader = DirectoryLoader(path=rc.get('DEFAULT','data_dir'),
                            glob=glob,
                            loader_cls=loader_cls,
                            silent_errors=False,    
                            loader_kwargs=loader_kwargs)
                docs = loader.load()
                if len(docs) >0:
                    if docs[0].page_content == '':
                        logging.error("No content in %s",docs[0].metadata['source'])    
                    else:
                        for doc in docs:
                            if dict(doc)['metadata']:
                                mtd = dict(doc)['metadata']
                                for key in mtd:
                                    if type(mtd[key]) == list:
                                        mtd[key] = ','.join([str(item) for item in mtd[key]])
                                doc.metadata = mtd
                        vectorstore.add_documents(docs)
                        logging.info("Context loaded from %s documents, %s docs",ftype, str(len(docs)))              


def initialize_chain(new_vectorstore=False):
    """ initialize the chain to access the LLM """

    this_model = globvars['LLM']
    collection_name="vectorstore"

    if not new_vectorstore and os.path.exists(rc.get('DEFAULT','persistence')+'/chroma.sqlite3'):
        persistent_client = chromadb.PersistentClient(
            path=rc.get('DEFAULT','persistence'))
        vectorstore = Chroma(client=persistent_client,
                             collection_name=collection_name,
                             collection_metadata={"hnsw:space": "cosine"},
                             embedding_function=embedding_function())
        globvars['NoChunks'] = len(vectorstore.get()['ids'])
        logging.info("Loaded %s chunks from persistent vectorstore", len(vectorstore.get()['ids']))
    else:
        persistent_client = chromadb.PersistentClient(
            path=rc.get('DEFAULT','persistence'))

        # Since we can't detect existence of a collection, we create one before deleting
        # Only needed in a start condition where no persitent store was found.
        # This way all fisimilarity_score_thresholdles are deleted compared to .reset which only disconnects.
        persistent_client.get_or_create_collection(name=collection_name)
        persistent_client.delete_collection(name=collection_name)
        vectorstore = Chroma(client=persistent_client,
                             collection_name=collection_name,
                             collection_metadata={"hnsw:space": "cosine"},
                             embedding_function=embedding_function())
        # Delete previous stored documents

        load_files(vectorstore, "all")
        logging.info("Stored %s chunks into vectorstore",len(vectorstore.get()['ids']))

    globvars['VectorStore'] = vectorstore
    globvars['NoChunks'] = len(vectorstore.get()['ids'])
    retriever = vectorstore.as_retriever(search_type=rc.get('DEFAULT','search_type'),
        search_kwargs={'score_threshold': rc.getfloat('DEFAULT','score'), 
                       'k':8})

    ### Contextualize question ###
    if rcmodel.startswith("o1"):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rcsysprompt1),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    else:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
    history_aware_retriever = create_history_aware_retriever(
        this_model, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = rcsysprompt2 + (
        "{context}"
    )
    if rcmodel.startswith("o1"):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    else:
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
    def log_retrieved_documents(retrieved_docs):
        for doc in retrieved_docs:
            logging.info("Retrieved document: %s", doc.page_content)

    question_answer_chain = create_stuff_documents_chain(llm=this_model, prompt=qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    globvars['Chain'] = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        callbacks=[log_retrieved_documents]
    )
    logging.info("Chain initialized: %s",globvars['ModelText'])
    return True

def create_call(name, function, methods, params=[], response_output="text"):
    """ Create Flask route """
    app_path = "/prompt/"+rc.get('DEFAULT','id')
    #@app.route(app_path+name, methods=methods)
    @cross_origin()
    def _function():
        logging.info("This call uses model %s",globvars['ModelText'])
        if len(request.files) > 0:
            values = request.files
        else:
            values = []
            for param in params:
                    if request.method == 'GET':
                        values.append(request.values[param])
                        logging.info("%s: %s", param, values[-1])
                    if request.method == 'POST':
                        values.append(request.form[param])
                        logging.info("%s: %s", param, values[-1])
        try: 
            result = function(values)
            match response_output:
                case "text":
                    logging.info("Result %s: %s",name, result['answer'])
                    return make_response(result['answer'], 200)
                case "context":
                    mylist = [item.to_json() for item in result['context']]
                    myanswer = result['answer']
                    mylist = pprint.pformat(object=[item['kwargs'] for item in mylist], width=132)
                    logging.info("Processing %s succeded",name)
                    return "\n".join([mylist,myanswer,""])
                case "json":
                    logging.info("Processing %s succeded",name)
                    mylist = {str(item):str(result[item]) for item in result}
                    return mylist
                case "file":
                    logging.info("Processing %s succeded",name)
                    return result
                case "search":
                    logging.info("Processing %s succeded",name)
                    mylist = [{'page_content':item[0].page_content,
                              'metadata':item[0].metadata,
                              'score':item[1]}
                              for item in result]
                    return mylist
                
        except HTTPException as e:
            logging.error("Error processing %s: %s",name, str(e))
            return make_response(f"Error processing {name}", 500)
    _function.__name__ = 'p'+name
    if name == '':
        my_path = app_path
    else:
        my_path = '/'.join([app_path,name])
    logging.info("path -> "+my_path+" "+','.join(params))
    app.add_url_rule(my_path, name, _function, methods=methods)

def log_error(error_text):
    logging.error(error_text)
    raise HTTPException(error_text)

def prompt(values):
    """ Answer the prompt """
    return globvars['Chain'].invoke(
            {"input": values[0]},
            config={"configurable": {"session_id": globvars['Session']}},
        )

create_call('', prompt, ["GET", "POST"], ['prompt'])
create_call('full', prompt, ["GET", "POST"], ['prompt'], "context")

def search(values):
    """ Search in vectorstore """
    sim = int(globvars['Similar'])
    if (len(values) == 2):
        sim = int(values[1])
    answer = globvars['VectorStore'].similarity_search_with_relevance_scores(
        values[0],
        sim)
    return answer

create_call('search', search, ["GET", "POST"], ['prompt','similar'],"search")

def documents(values):
    """ Get documents from vectorstore """
    id = values[0]
    if id != '':
        documents = globvars['VectorStore'].get(id)
    else:
        documents = globvars['VectorStore'].get()
    return documents

create_call('documents', documents, ["GET"], ['id'],"json")

def parameters(values):
    """ Return paramater values """
    parameter = values[1]
    section   = values[0]
    answer = rc.get(section=section,option=parameter)
    return {'answer':answer}

create_call('params', parameters, ["GET"], ['section','param'])

def globals(values):
    """ Return global values """
    return globvars

create_call('globals', globals, ["GET"], [], 'json')

def model_names(values):
    """ Return paramater values """
    answer = modelnames
    return {'answer':answer}

create_call('modelnames', model_names, ["GET"])

def embedding_names(values):
    """ Return paramater values """
    answer = embeddingnames
    return {'answer':answer}

create_call('embeddingnames', embedding_names, ["GET"])

def model(values):
    """ Set the LLM model """
    this_model = values[0]
    if not this_model.isascii():
        log_error("Modelname "+this_model+" uses non-ascii characters")
    if not this_model in modelnames:
        log_error("Model "+this_model+" not found in OpenAI's models")
    globvars['ModelText'] = this_model
    set_chat_model(globvars['Temperature'])
    initialize_chain(True)
    return {'answer':'Model set to '+this_model}
    
create_call('model', model, ["GET", "POST"], ['model'])

def embeddings(values):
    """ Set the Embedding model """
    this_embedding = values[0]
    if not this_embedding.isascii():
        log_error("Embedding name "+this_embedding+" uses non-ascii characters")
    if not this_embedding in embeddingnames:
        log_error("Embedding "+this_embedding+" not found in OpenAI's models")
    globvars['Embedding'] = this_embedding
    set_chat_model(globvars['Temperature'])
    initialize_chain(True)
    return {'answer':'Embedding set to '+this_embedding}
    
create_call('embeddings', embeddings, ["GET", "POST"], ['embedding'])

def chunk(values):
    """ Set the LLM model """
    this_chunk_size = int(values[0])
    this_chunk_overlap = int(values[1])
    if this_chunk_size < 1 or this_chunk_size > 1000:
        log_error("Chunk size "+str(this_chunk_size)+" not between 1 and 1000") 
    if this_chunk_overlap < 0 or this_chunk_overlap > 100:
        log_error("Chunk overlap "+str(this_chunk_overlap)+" not between 0 and 100")
    globvars['ChunkSize'] = this_chunk_size
    globvars['ChunkOverlap'] = this_chunk_overlap
    initialize_chain(True)
    return {'answer':'Chunk set to '+str(this_chunk_size)+' with overlap '+str(this_chunk_overlap)}
    
create_call('chunk', chunk, ["GET", "POST"], ['chunk_size','chunk_overlap'])

def temp(values):
    """ Set temperature """
    temperature = float(values[0])
    if temperature < 0.0 or temperature > 2.0:
        log_error("Temperature "+str(temperature)+" not between 0.0 and 2.0")
    globvars['Temperature'] = temperature
    set_chat_model(temperature) 
    initialize_chain(True)
    return {'answer':'Temperature set to '+str(temperature)}

create_call('temp', temp, ["GET", "POST"], ['temp'])

def reload(values):
    """ Reload documents """
    initialize_chain(True)
    return {'answer':'Documents reloaded'}

create_call('reload', reload, ["GET", "POST"])

def clear(values):
    """ Clear the cache """
    globvars['Store'].clear()
    return {'answer':'History deleted'}

create_call('clear', clear, ["GET", "POST"])


def cache(values):
    """ Return cache contents """
    content = ""
    if globvars['Session'] in globvars['Store']:
        for message in globvars['Store'][globvars['Session']].messages:
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"
            content += prefix +  ":" + str(message) + "\n"
    return {'answer': content}

create_call('cache', cache, ["GET", "POST"])

def send_files(values):
    """ Serve HTML files """
    file = values[0]
    html_dir = rc.get('DEFAULT','html')
    absolute_path = html_dir[0:1] == '/'
    if absolute_path:
        serve_file = os.path.normpath(os.path.join(html_dir,file))
    else:
        serve_file = os.path.normpath(os.path.join(base_dir + html_dir,file))
    if not serve_file.startswith(base_dir):
        logging.error("Parameter value for HTML not allowed")
    logging.info('File served: '+serve_file)
    return send_file(serve_file)

create_call('file', send_files, ["GET"], ['file'], "file")

def list_files(values):
    """ List context files """
    file = values[0]
    action = values[1]
    if action not in ['list','delete']:
        logging.error(f"Action {action} value not allowed, only list or delete")
    context_dir = rc.get('DEFAULT','data_dir')
    serve_files = os.path.normpath(os.path.join(base_dir,context_dir))
    if not serve_files.startswith(base_dir):
        logging.error("Parameter value for DATA_DIR not allowed")
    all_files = os.listdir(serve_files)
    output = {
        "name": "Context files",
        "type": "folder",
        "items": []
    }
    if action == 'delete':
        os.remove(os.path.join(serve_files,file))  
    context_files = [file for file in all_files if os.path.isfile(os.path.join(serve_files,file))]  
    logging.info('Context files: '+ ','.join(context_files))
    for file in context_files:
        output['items'].append(
            {
                "name": file,
                "type": "file",
            }
        )
    return output

create_call('context', list_files, ["GET"],['file','action'],"file")

def process_image(values):
    """ Send image to ChatGPT and send prompt to analyse contents """
    logging.info(globvars['ModelText'])
    my_url = values[0]
    text   = values[1]
    if not my_url.isalnum:
        log_error("URL is not well-formed")
    if globvars['ModelText'] != 'gpt-4o':
        log_error("Image processing only available in gpt-4o")

    image_url = quote(my_url, safe='/:?=&')
    logging.info("Processing image: %s, with prompt: %s", image_url, text)
    bimage = encode_image(image_url)
    chain = ChatOpenAI(api_key=os.environ.get('OPENAI_APIKEY'),
                       model=globvars['ModelText'],
                        temperature=globvars['Temperature'])
    msg = chain.invoke(
        [
            AIMessage(content="Picture revealer"),
            HumanMessage(content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{bimage}"}}
            ])
        ]
    )
    return {'answer': msg.content}

create_call('image', process_image, ["GET","POST"], ['image','prompt'])

def upload_file(values):
    """ Handles the file upload """

    file = values['file']
    if file.filename == '':
        log_error("Error in upload data, no filename found")

    filename = secure_filename(file.filename)

    # module 'constants' is not iterable, so repeating code unfortunately
    found = False
    if PurePath(filename).match(rc.get('DEFAULT','data_glob_txt')):
        found = True
    if PurePath(filename).match(rc.get('DEFAULT','data_glob_pdf')):
        found = True
    if PurePath(filename).match(rc.get('DEFAULT','data_glob_docx')):
        found = True
    if PurePath(filename).match(rc.get('DEFAULT','data_glob_xlsx')):
        found = True
    if PurePath(filename).match(rc.get('DEFAULT','data_glob_pptx')):
        found = True

    if not found:
        log_error("File to upload has extension that doesn't match GLOB_* constants")

    filepath = os.path.join(rc.get('DEFAULT','data_dir'),filename)

    logging.info("Saving %s on: %s",filename,filepath)
    file.save(filepath)
    logging.info("File %s saved on: %s",filename,filepath)
    load_files(globvars['VectorStore'], filename.split('.')[-1])
    return {'answer': 'Upload completed'}

create_call('upload', upload_file, ["POST"])
    
@app.teardown_request
def log_unhandled(e):
    if e is not None:
        print(repr(e))
        
if __name__ == '__main__':
    # Call an API to get the port number
    # fall back if config server is not available
    host = 'localhost'
    port = 5200
    try:
        response = urlopen(rcconfighost + "/get?project=" + rag_project)
        configs = json.loads(response.read().decode('utf-8'))
        port = int(configs['port'])
        logging.info("Port found at config server: %s", str(port))
    except Exception as e:
        logging.error("Failed to get port number from API: %s", e)
        logging.error("Defaulting to port nr: %s", str(port))
    if initialize_chain():
        app.run(port=port, debug=rc.getboolean('FLASK', 'debug'), host="0.0.0.0")
    else:
        raise Exception("Initialization of chain failed")

