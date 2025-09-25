import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import langchain_core.prompts as prompts
import langchain_core.output_parsers as StrOutputParser
import os


import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking & Google Gemini
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")  
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Google Gemini"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def generate_response(modelName, question,max_tokens):

    llm= ChatGoogleGenerativeAI(
        model=modelName,
        max_tokens=max_tokens
    )
    ## Prompt Template
    prompt=prompts.ChatPromptTemplate.from_messages(
        [
            ("system","You are a helpful assistant.please response to the user queries"),
            ("user","Question:{question}")
        ]
    )

    
    
    chain =prompt | llm 
    answer=chain.invoke({'question':question})
    

    return answer.content


#### streamlit app
st.title("Q&A Chatbot")


st.sidebar.title("Settings")

## Select the Google Gemini model
modelNames = st.sidebar.selectbox("Select Google Gemini model",["gemini-1.5-flash","gemini-1.5-pro","gemini-2.0"])  

## Adjust response parameter
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Go ahead and ask any question")

user_input=st.text_input("You:")

if user_input:
    response = generate_response(modelNames, user_input, max_tokens)
    st.write(response)
else:
    st.error("Please enter the question to get the answer")


    