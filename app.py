import streamlit as st
from llama_index.main import LlamaIndexInterface
#from lang_chain.main import LangChainInterface


TITLE="Maroc Telecom assistant"
ASSISTANT_MESSAGE = "Message IAM Assistance"
RAG=""

def response_generator(prompt):
    response = f"response for {prompt} ..."
    return response


# GUI logic #

st.title(TITLE)

# Initialize the RAG backend
if 'interface' not in st.session_state:
    st.session_state.interface = LlamaIndexInterface()

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
    response = st.session_state.interface.query(prompt=prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
