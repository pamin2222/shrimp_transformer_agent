import openai
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

import shrimp_helper
from key_helper import check_openai_key
import streamlit as st
from enum import Enum


class AIMode(Enum):
    FULL_SHRIMP_MODE = 'Full Shrimp Mode'
    PARTIAL_SHRIMP_MODE = 'Partial Shrimp Mode'
    NORMAL = 'Normal'


# Add input for OpenAI API key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    target_word_input = st.text_input(
        "Text input for AI to use",
        "Shrimp",
        key="placeholder",
    )
    ai_mode_selection = st.radio(
        "Set word replace mode 👉",
        key="ai_mode",
        options=["Full Shrimp Mode", "Partial Shrimp Mode", "Normal"],
    )

# Map the selected radio option to the enum
ai_mode_enum = {
    "Full Shrimp Mode": AIMode.FULL_SHRIMP_MODE,
    "Partial Shrimp Mode": AIMode.PARTIAL_SHRIMP_MODE,
    "Normal": AIMode.NORMAL,
}[ai_mode_selection]

st.title("Shrimp Transformer")
st.caption("")

# Set up memory
msgs = StreamlitChatMessageHistory(key="history")

check_openai_key(openai_api_key)

# Set up LLMs
llm_shrimp = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
llm_shrimp_memory = ConversationBufferMemory()

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if len(msgs.messages) == 0:
    init_msg = "Input something to start a conversation"
    st.chat_message("system").write(init_msg)


def generate_conversation(latest_response, ai_mode, st):
    llm_shrimp_conver_chain = ConversationChain(
        llm=llm_shrimp,
        verbose=True,
        memory=llm_shrimp_memory,
        prompt=shrimp_helper.shrimpify_prompt_template
    )

    user_w_params = shrimp_helper.create_user_input_with_params(mode=ai_mode_selection, user_prompt=latest_response,
                                                                target_word=target_word_input)

    ai_response = llm_shrimp_conver_chain.predict(input=user_w_params)
    if ai_mode == AIMode.FULL_SHRIMP_MODE:
        shrimpified_response = ' '.join(
            [target_word_input if target_word_input not in word else word for word in ai_response.split()])
        ai_response = shrimpified_response

    # Add the AI response to the conversation container
    msgs.add_ai_message(ai_response)

    # Display the AI response in the chat interface
    st.chat_message("ai").write(ai_response)

    # The function should return the AI's response instead of the latest response from the user
    return ai_response


if prompt := st.chat_input():
    openai.api_key = openai_api_key
    msgs.add_user_message(prompt)
    st.chat_message("user").write(prompt)
    latest_response = generate_conversation(prompt, ai_mode_selection, st)
    print("ai_mode:", ai_mode_selection)
