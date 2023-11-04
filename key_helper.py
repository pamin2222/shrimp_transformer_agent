import streamlit as st
import os

def check_openai_key(openai_api_key):
    if 'OPENAI_API_KEY' in os.environ:
        return
    elif not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        os.environ['OPENAI_API_KEY'] = openai_api_key