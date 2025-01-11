import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json
import os
from huggingface_hub import login

# Step 1: Authenticate Hugging Face API
API_KEY = "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"
login(API_KEY)
os.environ["HUGGINGFACE_HUB_TOKEN"] = API_KEY

# Step 2: Define LLM with constraints
llm = HuggingFaceHub(
    repo_id="google/gemma-1.1-2b-it",
    model_kwargs={
        "temperature": 0.2,
        "repetition_penalty": 1.1,
        "max_new_tokens": 200
    },
    huggingfacehub_api_token=API_KEY
)

# Step 3: Define Review Schema
class ReviewAnalysis(BaseModel):
    summary: str = Field(description="A brief summary of the customer review with maximum 3 lines")
    positives: list = Field(description="Positives mentioned by the customer, max 3 points")
    negatives: list = Field(description="Negatives mentioned by the customer, max 3 points")
    sentiment: str = Field(description="Overall sentiment of the review (positive, negative, or neutral)")
    emotions: list = Field(description="3-5 emotions expressed based on the sentiment")
    email: str = Field(description="Detailed email to the customer based on the sentiment")

# Initialize the output parser
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# Step 4: Define Prompt Template
prompt_text = """
Analyze the customer review below and provide a JSON response with the following structure:

{
  "summary": "A brief summary of the customer review with maximum 3 lines",
  "positives": ["Positives mentioned in the review, max 3 points"],
  "negatives": ["Negatives mentioned in the review, max 3 points"],
  "sentiment": "Overall sentiment of the review (positive, negative, or neutral)",
  "emotions": ["3-5 emotions expressed based on the sentiment"],
  "email": "Detailed email to the customer based on the sentiment"
}

Make sure to fill the fields correctly. Do not include any extra information or metadata. Provide only the required JSON content.
Customer Review: {review}
"""

# Step 5: Define a sample review
simple_review = "The product is good, but the delivery was late."

# Step 6: Chain Execution
prompt = PromptTemplate(
    template=prompt_text,
    input_variables=["review"]  # Only 'review' is the required input variable
)
st.write(prompt)
st.write(parser)

chain = (prompt | llm | parser)

# Step 7: Execute chain and display the result using Streamlit
try:
    response = chain.invoke({"review": simple_review})
    review_analysis = ReviewAnalysis(**response)
    st.write("Response:", review_analysis.dict())

except ValidationError as exc:
    st.error(f"Validation Error: {exc}")
except Exception as e:
    st.error(f"Error: {e}")