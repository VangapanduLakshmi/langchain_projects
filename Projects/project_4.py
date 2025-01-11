import streamlit as st
from langchain.chat_models import ChatHuggingFace
from langchain import HuggingFaceHub


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
if prompt:
    res_1 = res.invoke(prompt)
    res = res_1.content
    cleaned_output = (
        res.replace("<bos>", "")
        .replace("<start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .strip()
    )
    st.write(cleaned_output)
