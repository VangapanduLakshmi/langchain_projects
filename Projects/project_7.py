#PROMPT ENGINEERING WITH LANGCHAIN AND HUGGINGfACE

import streamlit as st
from pydantic import BaseModel, Field
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Your project logic here...


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

reviews = [f"""The bag itself feels amazing, using very thick material 
          and it is also nicely padded. Having waterbottle holders on both sides is very nice. 
          There is a front pocket and a main pocket which has many pockets to keep smaller things. 
          The zipper is a little hard to use but it looks very neat and flat when it is zipped. 
          It functions well as a laptop bag but Im using this as a school bag and it fits everything I need. 
          The bag straps are adjustable. They make it very easy to carry and it feels lightweight even with many things in it."""]

class ReviewAnalysis(BaseModel):
    summary : str = Field(description = " A brief summary of the customer review with maximum 3 lines")
    positives: list = Field(description = " A list showing positives mentioned by the customer in the review if any - max 3 points")
    negatives: list = Field(description =  "A list showing negatives mentioned by the customer in the review if any - max 3 points")
    sentiment : str = Field(description = "One word showing of the review - positive, negative or neutral")
    emotions : list = Field(description = "A listof 3-5 emotions expressed based on the sentiment")
    email : str = Field(description = "Detailed email to the customer based on the sentiment")

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt_text = """"
              Analyze the given customer review below and generate the response based on the instructions
              mentioned below in the format instructions.
              Also remember to write a detailed email resposne for the email field based on these conditions:
                - email should be addresseed to Dear Customer and signed with Service Agent
                - thank them if the review is positive or neutral
                - apologize if the review is negative
              Formate Instructions: {format_instructions}
              Review: {review}"""
            
prompt = PromptTemplate(template = prompt_text,
                        input_variables = ["review"],
                        partial_variables = {"format_instructions": parser.get_format_instructions()}, 
)
st.write(prompt)
chain = (prompt | llm | parser)

#reviews_formated = {"review": reviews[0]}

response = chain.invoke({"review": reviews[0]})

st.write(response)