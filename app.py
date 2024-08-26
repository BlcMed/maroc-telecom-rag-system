import streamlit as st
from src.llama_index_backend.agent import LlamaIndexAgent
#from src.lang_chain.main import LangChainInterface
from config.config import load_config


# Load the configuration
if 'config' not in st.session_state:
    config = load_config()
    st.session_state.config = config

MODEL_NAME = st.session_state.config.get('MODEL_NAME')
CHUNK_SIZE = st.session_state.config.get('CHUNK_SIZE')
CHUNK_OVERLAP = st.session_state.config.get('CHUNK_OVERLAP')
SIMILARITY_TOP_K = st.session_state.config.get('SIMILARITY_TOP_K')
SIMILARITY_CUTOFF = st.session_state.config.get('SIMILARITY_CUTOFF')
DATA_PATH = st.session_state.config.get('DATA_PATH')
VECTOR_STORE_PATH_LLAMA_INDEX = st.session_state.config.get('VECTOR_STORE_PATH_LLAMA_INDEX')
VECTOR_STORE_PATH_LLANG_CHAIN = st.session_state.config.get('VECTOR_STORE_PATH_LLANG_CHAIN')
TITLE = st.session_state.config.get('TITLE')
ASSISTANT_MESSAGE = st.session_state.config.get('ASSISTANT_MESSAGE')


st.title(TITLE)

# Initialize the RAG backend
if 'agent' not in st.session_state:
    st.session_state.agent = LlamaIndexAgent(
        model = MODEL_NAME,
        data_path=DATA_PATH,
        vector_store_path=VECTOR_STORE_PATH_LLAMA_INDEX,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        similarity_cutoff=SIMILARITY_CUTOFF,
        similarity_top_k=SIMILARITY_TOP_K
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
    # Get response
    response, source = st.session_state.agent.question(prompt=prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        #st.markdown(f"> **Info Source:**\n> {source}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
