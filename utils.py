# type: ignore
import os
from typing import TextIO

import openai
import pandas as pd
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
import re

#openai.api_key = st.secrets["OPENAI_API_KEY"]

# custom function to list steps to answer a question then write and execute code to answer the question
def ask_question(llm, prompt, df):
    
    # create initial prompt
    df_template_1 = """
    You are an expert data scientist, working with a pandas dataframe in Python. The name of the dataframe is `df`.
    If a package is needed, call `import` at the top of the code.

    Your task is to create a plan of steps to take to answer the question below. 
    Do not return blocks of code, only the plan of steps needed to get the answer. 

    This is the results of `print(df.head())`:
    {df_head}

    This is the results of `print(df.describe())`:
    {df_describe}

    This is the results of `print(df.dtypes)`:
    {df_dtypes}

    This is the results of `print(df.columns)`:
    {df_columns}

    Begin!

    Question: {input}
    Answer:
    """

    prompt_template_1 = PromptTemplate.from_template(df_template_1)

    final_prompt_1 = prompt_template_1.format(df_head = str(df.head().to_markdown()), 
                                        df_describe = str(df.describe().to_markdown()),
                                        df_dtypes = str(df.dtypes.to_markdown()),
                                        df_columns = str(df.columns),
                                        input = prompt)

    #print(final_prompt_1)

    # call model
    answer_1 = llm(final_prompt_1)

    formatted_answer_1 = f"""{answer_1.strip()}"""

    st.write("Steps to Answer Request:")
    st.write(formatted_answer_1)

    # create prompt to execute steps from the first prompt
    df_template_2 = """
    You are an expert data scientist, working with a pandas dataframe in Python. The name of the dataframe is `df`.
    If a package is needed, call `import` at the top of the code.

    Your task is to execute the list of steps below to answer the question below. Only write code to execute the steps, 
    inside of a python code block like ```python```, with comments to help understand what you are doing and what step this code is executing.
    If given a choice between two or more packages to use in a step, choose the first one. 
    Any output from the code should be formatted to work in a streamlit app. 
    If you need to print something in the python code, use `st.write()`.
    If you need to show a plot, use `st.pyplot(fig)`.

    This is the results of `print(df.head())`:
    {df_head}

    This is the results of `print(df.describe())`:
    {df_describe}

    This is the results of `print(df.dtypes)`:
    {df_dtypes}

    This is the results of `print(df.columns)`:
    {df_columns}

    Begin!

    Question: {input}
    Steps: {steps}
    Answer:
    """

    prompt_template_2 = PromptTemplate.from_template(df_template_2)

    final_prompt_2 = prompt_template_2.format(df_head = str(df.head().to_markdown()), 
                                        df_describe = str(df.describe().to_markdown()),
                                        df_dtypes = str(df.dtypes.to_markdown()),
                                        df_columns = str(df.columns),
                                        steps = formatted_answer_1,
                                        input = prompt)

    #print(final_prompt_2)

    # call model
    answer_2 = llm(final_prompt_2)

    formatted_answer_2 = f"""{answer_2.strip()}"""

    #st.write("Code to Execute Steps:")
    #st.write(formatted_answer_2, end='\n\n')

    code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    code = code_pattern.findall(formatted_answer_2)

    extracted_code = code[0].strip()

    st.write("running analysis...")

    exec(extracted_code)

    # code_button = st.button("Show Python Code")
    # if code_button:
    #     st.write("Code to Execute Steps:")
    #     st.write(formatted_answer_2)

    with st.expander("Show Python Code"):
        st.write("Code to Execute Steps:")
        st.write(formatted_answer_2)



def get_answer_csv(file: TextIO, query: str) -> str:

    # Load the CSV file as a Pandas dataframe
    df = pd.read_csv(file)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # connect to model
    llm = OpenAI(temperature=0, model_name = "gpt-3.5-turbo")

    # custom function call to ask question and get steps to answer
    ask_question(llm, query, df)
