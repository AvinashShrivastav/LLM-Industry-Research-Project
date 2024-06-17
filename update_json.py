import plotly.graph_objects as go
import random
import google.generativeai as genai
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI as llmOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import config


# Code to get information using Chat-gpt 3.5 Turbo + PDF knowledge
def gpt_model_with_knowledge(vector_db, user_input, prompt):
    docs = vector_db.similarity_search(user_input)
    docs = ""

    llm = llmOpenAI(openai_api_key=config.open_ai_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    template = prompt
    response_gpt = chain.run(input_documents=docs, question=template+user_input)
    return response_gpt

# Code to get information using Gemini model + PDF knowledge
def gemini_model_with_knowledge(vector_db, user_input, prompt):

    docs = vector_db.similarity_search(user_input)
    docs = " "

    genai.configure(api_key=config.gemini_key)

    model = genai.GenerativeModel(model_name="gemini-pro")

    template =  prompt
    response = model.generate_content(str(docs) + template + user_input)

    return str(response.text)

