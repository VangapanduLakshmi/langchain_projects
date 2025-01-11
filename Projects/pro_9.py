import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from langchain_core.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import json
import os
from huggingface_hub import login

# Step 1: Authenticate Hugging Face API
API_KEY = "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"  # Replace with your actual API key
login(API_KEY)
os.environ["HUGGINGFACE_HUB_TOKEN"] = API_KEY

# Step 2: Define LLM with constraints
llm = HuggingFaceEndpoint(
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
    positives: list[str] = Field(description="Positives mentioned by the customer, max 3 points")
    negatives: list[str] = Field(description="Negatives mentioned by the customer, max 3 points")
    sentiment: str = Field(description="Overall sentiment of the review (positive, negative, or neutral)")
    emotions: list[str] = Field(description="3-5 emotions expressed based on the sentiment")
    email: str = Field(description="A detailed email to the customer based on the sentiment")

# Step 4: Define Prompt Template
prompt_template = """
Analyze the customer review below and provide a JSON response **strictly adhering to the following format**. Do not include any extra text or metadata.

**JSON Format:**
```json
{{
    "summary": "A brief summary of the customer review with maximum 3 lines",
    "positives": ["Positive aspect 1", "Positive aspect 2", ...],
    "negatives": ["Negative aspect 1", "Negative aspect 2", ...],
    "sentiment": "Overall sentiment (positive, negative, or neutral)",
    "emotions": ["Emotion 1", "Emotion 2", ...],
    "email": "A detailed email to the customer based on the sentiment"
}}
{{review}}
"""
simple_review = "The product is good, but the delivery was late."

prompt = PromptTemplate(
template=prompt_template,
input_variables=["review"]
)

try:
    # Remove any leading/trailing whitespace and non-JSON characters
    response = response.strip()
    if response.startswith("```json"):
        response = response[len("```json"):].strip()
    if response.endswith("```"):
      response = response[:-len("```")].strip()
    response_json = json.loads(response)
    review_analysis = ReviewAnalysis(**response_json)
    st.write("Response:", review_analysis.dict())
except json.JSONDecodeError as e:
    st.error(f"JSON Decode Error: {e}. Raw response: {response}")
except ValidationError as exc:
    st.error(f"Pydantic Validation Error: {exc}. Parsed JSON: {response_json if 'response_json' in locals() else 'No JSON parsed'}")