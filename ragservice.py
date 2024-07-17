from urllib.request import urlopen
from urllib.parse import quote
from flask import Flask, make_response, jsonify, request
from flask_cors import CORS, cross_origin
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
import base64
import os
import sys
import uuid
import importlib
import config
from operator import itemgetter
from typing import List

print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")
PROJECT=sys.argv[1]
constants = importlib.import_module("constants.constants_"+PROJECT)
app = Flask(__name__)
CORS(app)

# Configureer logging
logging.basicConfig(level=logging.getLevelName(constants.LOGGING_LEVEL))

os.environ["OPENAI_API_KEY"] = config.APIKEY

llm = ChatOpenAI(model="gpt-4o", temperature=constants.TEMPERATURE)
index = None
chain = None
store = {}
session = uuid.uuid4()

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    global store
    prefix = ""
    if session_id not in store:
#        store[session_id] = ChatMessageHistory()
        store[session_id] = InMemoryHistory()
    for message in store[session_id].messages:
        if isinstance(message, AIMessage):
            prefix = "AI"
        else:
            prefix = "User"
        print(f"{prefix}: {message.content}\n")
    return store[session_id]


def encode_image(image_url):
    logging.info(f"Encoding image: {image_url}")
    with urlopen(image_url) as url:
        f = url.read()
        image = base64.b64encode(f).decode("utf-8")
    return image

def initialize_chain(model=llm):
    global index, chain, llm, store, session

    text_loader_kwargs={'autodetect_encoding': True}

    loader = DirectoryLoader(constants.DATA_DIR, glob=constants.DATA_GLOB, loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    print(len(docs))

    if 'LANGUAGE' in constants.__dict__:
        text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language[constants.LANGUAGE], chunk_size=constants.chunk_size, chunk_overlap=constants.chunk_overlap)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=constants.chunk_size, chunk_overlap=constants.chunk_overlap)
    splits = text_splitter.split_documents(docs)
    #for split in splits:
    #   print(split)
    #print("\n\n")
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
        llm, retriever, contextualize_q_prompt
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
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    logging.info(f"Chain initialized: {model.name}")
    chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

@app.route("/prompt", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID, methods=["GET", "POST"])
@cross_origin()
def process():
    global store, chain
    try:
        query = request.values['prompt']
        logging.info(f"Query: {query}")
        result = chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": session}},
        )
        logging.info(f"Result: {result['answer']}")
        #logging.info(f"Usage: {result}") 
        return make_response(result['answer'], 200)
    except Exception as e:
        logging.error(f"Error processing prompt: {str(e)}")
        return make_response("Error processing prompt", 500)

@app.route("/prompt/model", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/model", methods=["GET", "POST"])
@cross_origin()
def model():
    global chain
    try:
        model = request.values['model']
        newmodel = ChatOpenAI(model=model)
        chain = initialize_chain(newmodel)
        print(newmodel.model_name)
        return make_response("Model set to: {newmode.model_name}" , 200)
    except Exception as e:
        logging.error(f"Error setting model: {str(e)}")
        return make_response("Error setting model", 500)

@app.route("/prompt/reload", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/reload", methods=["GET", "POST"])
@cross_origin()
def reload():
    global chain
    try:
        initialize_chain()
        return make_response("Text reloaded", 200)
    except Exception as e:
        logging.error(f"Error reloading text: {str(e)}")
        return make_response("Error reloading text", 500)

@app.route("/prompt/clear", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/clear", methods=["GET", "POST"])
@cross_origin()
def clear():
    global chain,store
    store.clear()
    return make_response("History deleted", 200)

@app.route("/prompt/cache", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/cache", methods=["GET", "POST"])
@cross_origin()
def cache():
    global store, session
    content = ""
    if session  in store:
        for message in store[session].messages:
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"
            content += f"{prefix}: {message.content}\n"
    return make_response(content, 200)


@app.route("/prompt/image", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/image", methods=["GET", "POST"])
@cross_origin()
def process_image():
    try:
        image_url = quote(request.values['image'], safe='/:?=&')
        text = request.values['prompt']
        logging.info(f"Processing image: {image_url}, with prompt: {text}")
        bimage = encode_image(image_url)
        chain = ChatOpenAI(model="gpt-4o", temperature=constants.TEMPERATURE)
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
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return make_response("Error processing image", 500)


if __name__ == '__main__':
    initialize_chain()
    app.run(port=constants.PORT, debug=constants.DEBUG, host="0.0.0.0")
