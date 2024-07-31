# RAG for Your Own Documents

This script sets up a Flask web application that allows users to interact with a Retrieval-Augmented Generation (RAG) system for their own documents. The system uses OpenAI's language models to process and answer queries based on the documents provided. The application supports various functionalities such as changing the model, adjusting the temperature, reloading documents, and processing images.

## Dependencies

The script imports several libraries and modules:

- Standard libraries: `logging`, `os`, `sys`, `uuid`, `importlib`, `base64`
- Flask and related libraries: `Flask`, `make_response`, `request`, `HTTPException`, `CORS`, `cross_origin`
- LangChain and related libraries: `HumanMessage`, `AIMessage`, `create_retrieval_chain`, `create_history_aware_retriever`, `create_stuff_documents_chain`, `Chroma`, `DirectoryLoader`, `TextLoader`, `BaseChatMessageHistory`, `BaseMessage`, `BaseModel`, `Field`, `ChatPromptTemplate`, `MessagesPlaceholder`, `RunnableWithMessageHistory`, `ChatOpenAI`, `OpenAIEmbeddings`, `RecursiveCharacterTextSplitter`, `Language`
- Custom configuration: `config`

## Initialization

The script starts by printing the command-line arguments and loading the project-specific constants module. It then sets up the Flask application and configures logging.

```python
print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")
PROJECT=sys.argv[1]
constants = importlib.import_module("constants.constants_"+PROJECT)
app = Flask(__name__)
CORS(app)
```

## Context Processor

A context processor is defined to store global variables in a Flask-friendly way.

```python
@app.context_processor
def context_processor():
    """ Store the globals in a Flask way """
    return dict()
```

## Global Variables

Global variables are initialized, including the language model, temperature, and session ID.

```python
globvars = context_processor()
globvars['ModelText']   = "gpt-4o"
globvars['Temperature'] = float(constants.TEMPERATURE)
globvars['Chain']       = None
globvars['Store']       = {}
globvars['Session']     = uuid.uuid4()
globvars['LLM']         = ChatOpenAI(model=globvars['ModelText'], 
                                     temperature=globvars['Temperature'])
```

## In-Memory History

A class for in-memory chat message history is defined.

```python
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []
```

## Session History

A function to get the session history is defined.

```python
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
        print("%s: %s\n",prefix, message.content)
    return Store[session_id]
```

## Image Encoding

A function to encode images to base64 is defined.

```python
def encode_image(image_url) -> base64:
    """ Encode image to base64 """
    logging.info("Encoding image: %s",image_url)
    with urlopen(image_url) as url:
        f = url.read()
        image = base64.b64encode(f).decode("utf-8")
    return image
```

## Chain Initialization

A function to initialize the chain for accessing the language model is defined.

```python
def initialize_chain(thisModel=globvars['LLM']):
    """ initialize the chain to access the LLM """

    text_loader_kwargs={'autodetect_encoding': True}

    loader = DirectoryLoader(constants.DATA_DIR, 
                             glob=constants.DATA_GLOB, 
                             loader_cls=TextLoader,
                             loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    print(len(docs))

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
        thisModel, retriever, contextualize_q_prompt
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
    question_answer_chain = create_stuff_documents_chain(thisModel, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    logging.info("Chain initialized: %s",globvars['ModelText'])
    globvars['Chain'] = RunnableWithMessageHistory(
        rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
```

## Routes

Several routes are defined to handle different functionalities:

### Process Prompt

Handles the main prompt processing.

```python
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
```

### Change Model

Handles changing the language model.

```python
@app.route("/prompt/model", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/model", methods=["GET", "POST"])
@cross_origin()
def model() -> make_response:
    """ Change LLM model """
    try:
        globvars['ModelText'] = request.values['model']
        globvars['LLM'] = ChatOpenAI(model=globvars['ModelText'], 
                                     temperature=globvars['Temperature'])
        initialize_chain(globvars['LLM'])
        logging.info("Model set to: %s", globvars['ModelText'])
        return make_response("Model set to: " + globvars['ModelText'] , 200)
    except HTTPException as e:
        logging.error("Error setting model: %s", str(e))
        return make_response("Error setting model", 500)
```

### Change Temperature

Handles changing the temperature of the language model.

```python
@app.route("/prompt/temp", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/temp", methods=["GET", "POST"])
@cross_origin()
def temp() -> make_response:
    """ Change LLM temperature """
    try:
        globvars['Temperature'] = float(request.values['temp'])
        globvars['LLM'] = ChatOpenAI(model=globvars['ModelText'], 
                                     temperature=globvars['Temperature'])
        initialize_chain(globvars['LLM'])
        logging.info("Temperature set to %s", str(globvars['Temperature']))
        return make_response("Temperature set to: " + str(globvars['Temperature']) , 200)
    except HTTPException as e:
        logging.error("Error setting temperature %s", str(e))
        return make_response("Error setting temperature", 500)
```

### Reload Documents

Handles reloading documents into the chain.

```python
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
```

### Clear Cache

Handles clearing the cache.

```python
@app.route("/prompt/clear", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/clear", methods=["GET", "POST"])
@cross_origin()
def clear() -> make_response:
    """ Clear the cache """
    globvars['Store'].clear()
    return make_response("History deleted", 200)
```

### Return Cache Contents

Handles returning the contents of the cache.

```python
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
```

### Process Image

Handles processing an image and sending it to ChatGPT for analysis.

```python
@app.route("/prompt/image", methods=["GET", "POST"])
@app.route("/prompt/"+constants.ID+"/image", methods=["GET", "POST"])
@cross_origin()
def process_image() -> make_response:
    """ Send image to ChatGPT and send prompt to analyse contents """
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
```

## Main

The script initializes the chain and starts the Flask application.

```python
if __name__ == '__main__':
    initialize_chain()
    app.run(port=constants.PORT, debug=constants.DEBUG, host="0.0.0.0")
```

This script provides a comprehensive setup for a RAG system using Flask and OpenAI's language models, allowing for various functionalities such as model and temperature adjustments, document reloading, and image processing.