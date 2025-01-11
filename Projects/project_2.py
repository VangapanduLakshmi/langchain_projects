import transformers
import streamlit as st
from langchain import HuggingFaceHub

from langchain_community.chat_models import ChatHuggingFace

API_KEY = "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"

llm = HuggingFaceHub(
       repo_id = "google/gemma-1.1-2b-it",
       model_kwargs = {"temparature":0.5 , "repitition_penalty":1.02, "max_new_tokens": 200, "max_length":190},
       huggingfacehub_api_token = API_KEY

)

chat_gamma = ChatHuggingFace( model_id = "google/gemma-1.1-2b-it", llm = llm)

prompt = st.chat_input("say some thing")

if prompt:
    messages = [{"role": "user", "content": prompt}]
    res = chat_gamma.invoke(messages)
    st.write(res)