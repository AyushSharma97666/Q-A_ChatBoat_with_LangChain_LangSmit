import streamlit as st
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv
load_dotenv()

# --- Load API Keys ---
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")



# --- Initialize Google Gemini LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0.2
)

# --- Initialize HuggingFace Embeddings ---
# Example: all-MiniLM-L6-v2 (fast and light)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", """
        Answer the question based on the context below.
        Please provide the most accurate response based on the question.

        <context>
        {context}
        </context>

        Question: {input}
    """)
])


def create_vectorStore():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embedding_model
        st.session_state.loader = PyPDFDirectoryLoader(r"pdfs")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200 )
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.documents[:5])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)



#streamlit app 

st.title("Conversational RAG With PDF uploads and chat history")


with st.sidebar :
    st.title("Embedding and LLM Model Selection")
    if st.button("Document Embedding"):
        create_vectorStore()
        st.write("Vector Database is ready")      

user_prompt = st.text_input("Enter your question here:")

if st.button("Get Answer") :
    if user_prompt:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)

        start=time.process_time()
        response=retrieval_chain.invoke({'input':user_prompt})
        print(f"Response time :{time.process_time()-start}")

        st.write(response['answer'])

        ## With a streamlit expander
        with st.expander("Document similarity Search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
