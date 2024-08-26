import streamlit as st
from src.llama_index_backend.agent import LlamaIndexAgent
#from src.lang_chain.main import LangChainInterface
from config.config import load_config

#TITLE="Maroc Telecom assistant"
#ASSISTANT_MESSAGE = "Message IAM Assistance"

def response_generator(prompt):
    response = f"response for {prompt} ..."
    return response


###   GUI logic   ###


# Load the configuration
if 'config' not in st.session_state:
    config = load_config()
    st.session_state.config = config

    MODEL_NAME = config.get('MODEL_NAME')
    DATA_PATH = config.get('DATA_PATH')
    VECTOR_STORE_PATH_LLAMA_INDEX = config.get('VECTOR_STORE_PATH_LLAMA_INDEX')
    VECTOR_STORE_PATH_LLANG_CHAIN = config.get('VECTOR_STORE_PATH_LLANG_CHAIN')
    TITLE = config.get('TITLE')
    ASSISTANT_MESSAGE = config.get('ASSISTANT_MESSAGE')


st.title(TITLE)

# Initialize the RAG backend
if 'interface' not in st.session_state:
    st.session_state.interface = LlamaIndexAgent(
        model = MODEL_NAME,
        data_path=DATA_PATH,
        vector_store_path=VECTOR_STORE_PATH_LLAMA_INDEX
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(ASSISTANT_MESSAGE)

if prompt :
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    #response = response_generator(prompt)
    response = st.session_state.interface.question(prompt=prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
