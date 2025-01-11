import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
#from langchain.memory import ConversationWindow
from langchain_community.chat_models import ChatHuggingFace

API_KEY =  "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"
llm = HuggingFaceHub(
       repo_id = "google/gemma-1.1-2b-it",
       model_kwargs = {"temparature":0.5 , "repitition_penalty":1.02, "max_new_tokens": 200, "max_length":190},
       huggingfacehub_api_token = API_KEY

)

#memory = ConversationBufferMemory()

#chain = ConversationChain( llm = llm , memory = memory)

prompt = st.chat_input("say something")

if prompt:
       res = llm.invoke(prompt)
       st.write(res)
#with st.expander("Conversation History"):
    #st.write(memory.chat_memory.messages)

