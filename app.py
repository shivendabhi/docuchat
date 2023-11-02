import openai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import tiktoken

st.title("DocuChat")

openai.api_key = st.secrets["OPENAI_API_KEY"]

pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
if pdf is not None:
    pdf_reader = PdfReader(pdf)
        
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)
 
        # # embeddings
    store_name = pdf.name[:-4]
    st.write(f'{store_name}')
    # st.write(chunks)
 
    if os.path.exists(f"{store_name}.pkl"):
         with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
 

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display "system" messages
        if message["role"] == "user" or message["role"] == "assistant":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    search_results = VectorStore.similarity_search(query=prompt, k=3)
    docs = [result.page_content for result in search_results]
    combined_content = "\n".join(docs) + "\n" + prompt

    # Add the combined content to the messages as a "system" message
    st.session_state.messages.append({"role": "system", "content": combined_content})

    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't display "system" messages
            if message["role"] == "user" or message["role"] == "assistant":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    