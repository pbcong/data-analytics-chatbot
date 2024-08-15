import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from utils import process_query, show_chat_history

dotenv.load_dotenv(".env", override=True)

st.session_state['llm'] = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.environ['OPEN_AI_API'])
greetings = '''Hello my name is Bin, I am your data analytic assistant. Feel free to ask me anything about your dataset'''
if 'conversations' not in st.session_state:
    st.session_state['conversations'] = [({'role':"System", 'content':"You are an assistant for a data analyst, try to help them with what they do"}, 0)]
    st.session_state['conversations'].append(({'role':'AI', 'content':greetings}, 0))
    st.session_state['conversations_text'] = [SystemMessage(content = "You are an assistant for a data analyst, try to help them with what they do")]
    st.session_state['conversations_text'].append(AIMessage(content=greetings))
st.title("Data analytics app")

with st.sidebar:
    df = st.file_uploader("Upload a file", type = ['csv'])


if df or 'df' in st.session_state:
    if df:
        st.session_state['df'] = pd.read_csv(df) 
    st.table(st.session_state['df'].head())
    st.session_state['agent'] = create_pandas_dataframe_agent(llm=st.session_state['llm'], agent_type='tool-calling', df=st.session_state['df'], verbose=True, allow_dangerous_code=True, return_intermediate_steps=True)

    if 'conversations' in st.session_state:
        # print(st.session_state['conversations'])
        show_chat_history()
        

    if query := st.chat_input("Your Message"):
        st.chat_message("Human").markdown(query)
        st.session_state['conversations'].append(({'role': 'Human', 'content':query}, 0))
        st.session_state['conversations_text'].append(HumanMessage(content=query))

        process_query(st.session_state.agent, query)

        # response = st.session_state['llm'].invoke(st.session_state['conversations'])
        #st.chat_message("AI").markdown(response)
        #st.session_state['conversations'].append(response)
        #st.markdown(st.session_state['conversations'])
else: 
    st.markdown("### Please upload your dataset")