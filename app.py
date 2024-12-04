import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from dotenv import load_dotenv

load_dotenv()

# Load the API keys
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_PROJECT"] = "QAChatbot_project_With_OLLAMA"

# prompt template

prompt = ChatPromptTemplate(

    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user", "Qusetion: {question}")
    ]
)

def generate_response(question, engine, temperature, max_tokens):
   
    llm = Ollama(model = engine)
    output_parser=StrOutputParser()
    chain = prompt | llm | output_parser

    answer = chain.invoke(
        {
            "question": question
        }
    )
    return answer

    
## Creation of app

st.title("Enhanced Q$A ChatBot With Ollama Open Source Model Gemma:2b")


#Drop down to select various OpenAI Models
engine = st.sidebar.selectbox("Select an Ollama Model", ["gemma:2b", "mistral", "llama3.1"])

# Adjust response parameter

temperature = st.sidebar.slider("Temperature", min_value = 0.0, max_value = 1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value=150)

## Main interface for user interface

st.write("Go Ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")




