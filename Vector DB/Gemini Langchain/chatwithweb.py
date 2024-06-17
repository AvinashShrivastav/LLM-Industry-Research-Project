import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# for import 3rd party in langchain for bs
from langchain_community.document_loaders import WebBaseLoader

# for  text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#3rd party tools for langchain import by using lanchain community
from langchain_community.vectorstores import Chroma # chroma is vectoer DB

#for embedding the text chunk use openai embedding model
from langchain_openai import OpenAIEmbeddings
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
# GOOGLE_API_KEY=userdata.get('AIzaSyBg-EkoWxKK-JmtiEfsNFqnKOn70vRTqBk')
# os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()

def get_response(user_input):
    return "Good question"

def get_vectorstore_from_url(url):
    #debug get the text in document form
    loader = WebBaseLoader(url)
    document =loader.load()

    #split the documnet into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = Chroma.from_documents(document_chunks, embedding = embeddings)
    return vector_store

#create retrivel chain
def get_context_retriver_chain(vector_store):
    llm = ChatOpenAI()

    retriver = vector_store.as_retriver()
    
    prompt = ChatPromptTemplate.from_messagesfrom_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    


# app config
st.set_page_config(page_title="Chat with website ",page_icon="BM")
st.title("Chat with Website")

if "chat_history" not in st.session_state:
    st.session_state.chat_history =[
        AIMessage(content="Hell I am Bot, How can i help you?"),
    ]

# add Side Bar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website url")   

if website_url is None or website_url =="":
    st.info("Please enter a website Url")
else:
    # for debug 
    document_chunks = get_vectorstore_from_url(website_url)
    with st.sidebar:
        st.write(document_chunks)
    #user input
    user_query = st.chat_input("Write your input here...")
    if user_query is not None and user_query!="":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        #test the retriver chain 
        retriver_documents = retriver_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })

        st.write(retriver_documents)
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
