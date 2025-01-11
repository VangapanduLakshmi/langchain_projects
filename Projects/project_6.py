from langchain.prompts import PromptTemplate
import streamlit as st
from langchain import HuggingFaceHub
from langchain.chat_models import ChatHuggingFace
from getpass import getpass
from langchain_core.messages import HumanMessage , SystemMessage

API_KEY = "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"
from huggingface_hub import login
login(API_KEY)
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] =  API_KEY

llm = HuggingFaceHub(
    repo_id = "google/gemma-1.1-2b-it", 
    model_kwargs = {"temparature": 0.5 , "repitation_penalty": 1.03, "max_new_tokens":200 , "max_length":180},
    huggingfacehub_api_token = API_KEY
)

prompt = st.chat_input("say something")


res = ChatHuggingFace(llm = llm)

messages = []
if prompt:
    prompt_temp = PromptTemplate.from_template(prompt)
    messages.append(HumanMessage(content = prompt_temp.format()))
    res_1 = res.invoke(messages)
    messages.append(res_1)
    res_2 = res_1.content
    cleaned_output = (
        res_2.replace("<bos>", "")
        .replace("<start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )
    parts = cleaned_output.split("model", 1) 
    formatted_output = f"user: {parts[0].strip()}"
    out_put_2 = f"model: {parts[1].strip()}"

    st.write(formatted_output)
    st.write(out_put_2)
    
