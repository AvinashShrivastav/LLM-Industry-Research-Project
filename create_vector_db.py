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
from langchain_community.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import config
# Code to create vector database using FIASS
def convert_pdf_to_vector_db(pdf_paths):
    return " "

    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for i, page in enumerate(pdf_reader.pages):
            if i >= 150:
                break
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=config.open_ai_key)
    return FAISS.from_texts(chunks, embeddings)
