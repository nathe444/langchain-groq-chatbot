import os
import streamlit as st
from langchain import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "give responses to the user queries"),
        ("user", "question:{question}")
    ]
)

st.title('LangChain Chat with GROQ API')
input_text = st.text_input('Waiting to share my wisdom...')


llm = ChatGroq(temperature=0.5)

output_parser = StrOutputParser()


chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)


if input_text:
    response = chain.run({'question': input_text})
    st.write(f"Chatbot: {response}")
