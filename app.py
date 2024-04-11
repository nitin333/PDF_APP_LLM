import streamlit as st
import os
from groq import Groq
import random
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import streamlit as st
#from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from google.oauth2 import id_token
from google.auth.transport import requests


def creds_entered():
    if st.session_state["user"].strip() == "admin" and st.session_state["passwd"].strip() == "admin":
        st.session_state["authenticated"] = True
    else :
            st.session_state["authenticated"] = False
            if not st.session_state["passwd"]:
                st.warning("Please enter Password.")
            elif not st.session_state["user"]:
                st.warning("Please enter username.")    
            else :
                 st.error("Invalid Username/Password :face_with_raised_eyebrow:")
                 

def authenticate_user():
    if "authenticated" not in st.session_state:
        st.text_input(label='Username :',value="",key="user",on_change=creds_entered)
        st.text_input(label='Password :',value="",key="passwd",type="password",on_change=creds_entered)
        return False
    else:
        if st.session_state["authenticated"]:
            return True
        else :  
            
            st.text_input(label='Username :',value="",key="user",on_change=creds_entered)
            st.text_input(label='Password :',value="",key="passwd",type="password",on_change=creds_entered)
            return False

if authenticate_user():                                 

        def get_pdf_text(pdf_docs):
            text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text 

        def get_text_chunks(text):
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            return chunks

        def get_vectorstore(text_chunks):
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small',
                                        openai_api_key="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")
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore

        def get_conversation_chain(vectorstore, model):
            llm = ChatGroq(
                    model_name=model,groq_api_key = "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"
            )
            memory = ConversationBufferMemory(
                memory_key='chat_history', return_messages=True)
            
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
            return conversation_chain

        def handle_userinput(user_question):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)



        def main():

            st.set_page_config(page_title="Chat with PDF", page_icon=":books")
            st.write(css, unsafe_allow_html = True)
        
            if "conversation" not in st.session_state:
                st.session_state.conversation = None
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = None

            st.header("Chat with multiple PDFs :books:")

            user_question = st.text_input("Ask a question about your document:")
            
            if user_question:
                handle_userinput(user_question)

            with st.sidebar:
                st.title('Customization')
                
                model = st.sidebar.selectbox(
                'Choose a model',
                ['mixtral-8x7b-32768', 'llama2-70b-4096'])
                
                # conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)
                
                # memory=ConversationBufferWindowMemory(k=conversational_memory_length)

                st.subheader("Your Documents")
                
                pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
                
                if st.button("Process"):
                    with st.spinner("Processing"):
                        raw_text = get_pdf_text(pdf_docs)
                        
                        text_chunks = get_text_chunks(raw_text)

                        vectorstore = get_vectorstore(text_chunks)

                        st.session_state.conversation = get_conversation_chain(vectorstore, model)
                        st.write(f"Selected model: {model}")
        if __name__ == '__main__':
            main()
