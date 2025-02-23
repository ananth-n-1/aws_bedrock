import boto3
import json
import os 
import sys
import streamlit as st

# For Embedding 

from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data Ingesion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vectore store

from langchain_community.vectorstores import FAISS

# LLMS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

bedrock = boto3.client(service_name='bedrock-runtime')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

#data ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splliter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splliter.split_documents(documents)
    return docs

def get_vectore_store(docs):
    vectore_store = FAISS.from_documents(docs,
                                         bedrock_embeddings)
    vectore_store.save_local('faiss_index')

def get_llama():
    model = Bedrock(model_id="meta.llama3-70b-instruct-v1:0",
                    client=bedrock,
                    model_kwargs={'max_gen_len':512})
    return model

prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])

def response_llm(llm, vectore_store, query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectore_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer = qa({'query':query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title('create or update vectore store.')
        if st.button('update vectore store'):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vectore_store(docs)
                st.success('Done')
    if st.button('output'):
        vc_store = FAISS.load_local('faiss_index', 
                                    bedrock_embeddings,
                                    allow_dangerous_deserialization=True)
        llm = get_llama()
        answer = response_llm(llm, vc_store, user_question)

        st.write(answer)
        st.success('Done')

if __name__=="__main__":
    main()






