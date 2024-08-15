""" Example constants file, to be imported from <your project dir>/constants """

# simple string like "mydata"
ID="<your identifier"
# any unused port, will run the Flask server
PORT=7100
# Set to True if you want to debug
DEBUG=False
# Any other level will make it less verbose
LOGGING_LEVEL="INFO"
# The LLM to be used as listed in OpenAI
MODELTEXT="gpt-4o"
# Persistence directory for vectorstore
PERSISTENCE="data/_unittest/vectorstore"
# Directory that will be scanned for files to be added to the context
DATA_DIR="/home/data"
# All the file extentions you want to be part of the context, see LangChain documentation
DATA_GLOB_TXT="*.txt"
DATA_GLOB_PDF="*.pdf"
# Where the HTML files reside
HTML="/home/data/_unittest/html"
# the lower, the more precise. TODO can be changed dynamically
TEMPERATURE=0.0
# Influences the way answers are produced
contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
# Influences the way answers are produced
system_prompt = ( # choose your language, English works best
        "You are a chatbot"
    )
chunk_size=1000    # depending on your data
chunk_overlap=100  # idem
