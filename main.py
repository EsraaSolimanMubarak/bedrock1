from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
import os
import streamlit as st

# os.environ["AWS_PROFILE"] = "amplify-react"

modelID = "anthropic.claude-v2"

llm = BedrockLLM(
    model_id=modelID,
    region_name="us-east-1",
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.9}
)

def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. You are in {language}.\n\n{freeform_text}"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'freeform_text': freeform_text})
    return response

st.title("Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["english", "arabic"])

if language:
    freeform_text = st.sidebar.text_area(
        label="What is your question?",
        max_chars=100
    )

if freeform_text:
    response = my_chatbot(language, freeform_text)
    st.write(response['text'])
