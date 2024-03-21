import streamlit as st
import os
import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
KEY=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-3.5-turbo", temperature=0.5)
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""


TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

# Define functions for quiz generation and evaluation
def generate_quiz(text, number, subject, tone, response_json):
    quiz_generation_prompt = PromptTemplate(
        input_variables=["text", "number", "subject", "tone", "response_json"],
        template=TEMPLATE
    )
    quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
    response = quiz_chain(
        {
            "text": text,
            "number": number,
            "subject": subject,
            "tone": tone,
            "response_json": response_json
        }
    )
    return response["quiz"]

def evaluate_quiz(subject, quiz):
    quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE2)
    review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)
    response = review_chain(
        {
            "subject": subject,
            "quiz": quiz
        }
    )
    return response["review"]

# Streamlit app
st.title("MCQ Generator and Evaluator")

# File upload
uploaded_file = st.file_uploader("Upload your file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
else:
    text = ""

# User inputs
subject = st.text_input("Enter the subject:")
number = st.number_input("Enter the total number of MCQs:", min_value=1, value=5, step=1)
tone = st.selectbox("Select the tone:", ["simple", "formal", "casual"])

# Generate quiz
if st.button("Generate MCQs"):
    response_json = json.dumps(RESPONSE_JSON)
    quiz = generate_quiz(text, number, subject, tone, response_json)
    quiz = json.loads(quiz)
    
    # Display generated MCQs
    st.subheader("Generated MCQs:")
    for key, value in quiz.items():
        st.write(f"**{value['mcq']}**")
        for option, option_value in value["options"].items():
            st.write(f"{option}: {option_value}")
        st.write(f"Correct: {value['correct']}")
        st.write("")

    # Evaluate quiz
    st.subheader("Quiz Evaluation:")
    review = evaluate_quiz(subject, quiz)
    st.write(review)
