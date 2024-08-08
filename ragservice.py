"""_summary_
RAG for your own documents
"""
import logging
import os
import sys
import uuid
import importlib
import base64
from typing import List
from urllib.request import urlopen
from urllib.parse import quote
from flask import Flask, make_response, request
from werkzeug.exceptions import HTTPException
from flask_cors import CORS, cross_origin
import openai
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

import config

print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")
PROJECT=sys.argv[1]
constants_import = "constants.constants_"+PROJECT

constants = importlib.import_module(constants_import)
app = Flask(__name__)
CORS(app)

@app.context_processor
def context_processor():
    """ Store the globals in a Fals way """
    return dict()

# Configureer logging
logging.basicConfig(level=logging.getLevelName(constants.LOGGING_LEVEL))

os.environ["OPENAI_API_KEY"] = config.APIKEY
client     = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
models     = client.models.list().to_dict()['data']
modelnames = []
for modelitem in models:
    modelnames.append(modelitem['id'])

globvars = context_processor()
globvars['ModelText']   = constants.MODELTEXT
globvars['Temperature'] = float(constants.TEMPERATURE)
globvars['Chain']       = None
globvars['Store']       = {}
globvars['Session']     = uuid.uuid4()
globvars['LLM']         = ChatOpenAI(model=globvars['ModelText'],
                                     temperature=globvars['Temperature'])


def check_model_existence(modelText) -> bool:
    """ chack if model is present in OpenAI's models """
    if modelText in modelnames:
        return True
    else:
        return False
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

def initialize_chain():
    """ initialize the chain to access the LLM """

    if not check_model_existence(globvars['ModelText']):
        logging.error("Model %s not found in OpenAI's models",globvars['ModelText'])
        return False

    this_model = globvars['LLM']
    text_loader_kwargs={'autodetect_encoding': True}

    loader = DirectoryLoader(constants.DATA_DIR,
                             glob=constants.DATA_GLOB,
                             loader_cls=TextLoader,
                             loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    logging.info("Context loaded from %s documents",str(len(docs)))

    if 'LANGUAGE' in constants.__dict__:
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language[constants.LANGUAGE],
            chunk_size=constants.chunk_size,
            chunk_overlap=constants.chunk_overlap)
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=constants.chunk_size,
            chunk_overlap=constants.chunk_overlap)

    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    ### Contextualize question ###
    contextualize_q_system_prompt = constants.contextualize_q_system_prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        this_model, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    system_prompt = constants.system_prompt + (
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(this_model, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    globvars['Chain'] = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    logging.info("Chain initialized: %s",globvars['ModelText'])
    return True

@app.route("/prompt", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID, methods=["GET", "POST"])
@cross_origin()
def process() -> make_response:
    """ The prompt processor """
    try:
        query = request.values['prompt']
        logging.info("Query: %s", query)
        result = globvars['Chain'].invoke(
            {"input": query},
            config={"configurable": {"session_id": globvars['Session']}},
        )
        logging.info("Result: %s",result['answer'])
        return make_response(result['answer'], 200)
    except HTTPException as e:
        logging.error("Error processing prompt: %s",str(e))
        return make_response("Error processing prompt", 500)

@app.route("/prompt/model", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/model", methods=["GET", "POST"])
@cross_origin()
def model() -> make_response:
    """ Change LLM model """
    try:
        this_model = request.values['model']
        if not check_model_existence(this_model):
            error_text = "Model "+this_model+" not found in OpenAI's models"
            logging.error(error_text)
            return make_response(error_text, 500)

        globvars['ModelText'] = this_model
        globvars['LLM'] = ChatOpenAI(model=globvars['ModelText'],
                                     temperature=globvars['Temperature'])
        initialize_chain()
        error_text = "Model set to: " + globvars['ModelText']
        logging.info(error_text)
        return make_response( error_text, 200)
    except HTTPException as e:
        logging.error("Error setting model: %s", str(e))
        return make_response("Error setting model", 500)

@app.route("/prompt/temp", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/temp", methods=["GET", "POST"])
@cross_origin()
def temp() -> make_response:
    """ Change LLM temperature """
    try:
        temperature = float(request.values['temp'])
        if temperature < 0.0 or temperature > 2.0:
            raise HTTPException
        globvars['Temperature'] = temperature
        globvars['LLM'] = ChatOpenAI(model=globvars['ModelText'],
                                     temperature=globvars['Temperature'])
        initialize_chain()
        logging.info("Temperature set to %s", str(globvars['Temperature']))
        return make_response("Temperature set to: " + str(globvars['Temperature']) , 200)
    except HTTPException as e:
        logging.error("Error setting temperature %s", str(e))
        return make_response("Error setting temperature", 500)


@app.route("/prompt/reload", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/reload", methods=["GET", "POST"])
@cross_origin()
def reload() -> make_response:
    """ Reload documents to the chain """
    try:
        initialize_chain()
        return make_response("Text reloaded", 200)
    except HTTPException as e:
        logging.error("Error reloading text: %s" ,str(e))
        return make_response("Error reloading text", 500)


@app.route("/prompt/clear", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/clear", methods=["GET", "POST"])
@cross_origin()
def clear() -> make_response:
    """ Clear the cache """
    globvars['Store'].clear()
    return make_response("History deleted", 200)

@app.route("/prompt/cache", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/cache", methods=["GET", "POST"])
@cross_origin()
def cache() -> make_response:
    """ Return cache contents """
    content = ""
    if globvars['Session'] in globvars['Store']:
        for message in globvars['Store'][globvars['Session']].messages:
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"
            content += prefix +  ":" + str(message) + "\n"
    return make_response(content, 200)


@app.route("/prompt/image", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/image", methods=["GET", "POST"])
@cross_origin()
def process_image() -> make_response:
    """ Send image to ChatGPT and send prompt to analyse contents """
    logging.info(globvars['ModelText'])
    if globvars['ModelText'] != 'gpt-4o':
        return make_response("Image processing only available in gpt-4o", 500)
    try:
        image_url = quote(request.values['image'], safe='/:?=&')
        text = request.values['prompt']
        logging.info("Processing image: %s, with prompt: %s", image_url, text)
        bimage = encode_image(image_url)
        chain = ChatOpenAI(model=globvars['ModelText'],
                           temperature=globvars['Temperature'])
        msg = chain.invoke(
            [
                AIMessage(content="Gerco's foto ontdekker"),
                HumanMessage(content=[
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{bimage}"}}
                ])
            ]
        )
        return make_response(msg.content, 200)
    except HTTPException as e:
        logging.error("Error processing image: %s", str(e))
        return make_response("Error processing image", 500)

if __name__ == '__main__':
    if initialize_chain():
        app.run(port=constants.PORT, debug=constants.DEBUG, host="0.0.0.0")
    else:
        logging.error("Initialization of chain failed")
