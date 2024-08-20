import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage


# for any answers that contains graphs, chat log is saved as (text, graph==0 if no plot)
def exec_code(code: str, df: pd.DataFrame):
    try:
        local_vars = {"plt": plt, "df": df}
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals(), local_vars)

        return plt.gcf()
    except Exception as e:
        st.error(f"Error executing code: {e}")
        return None


def process_query(agent, query):
    response = agent(st.session_state["conversations_text"])
    try:
        code = response["intermediate_steps"][-1][0].tool_input["query"]
    except Exception:
        code = ""
        pass
    if "plt" in code:
        fig = exec_code(code, st.session_state["df"])
        with st.chat_message("AI"):
            st.write(response["output"])
            st.code(code)
            if fig:
                st.pyplot(fig)

        
        st.session_state.conversations.append(
            (
                {
                    "role": "AI",
                    "content": response["output"] + f"\n```python\n{code}\n```",
                },
                fig,
            )
        )
        st.session_state.conversations_text.append(
            AIMessage(content=response["output"] + f"\n```python\n{code}\n```")
        )

    else:
        with st.chat_message("AI"):
            st.write(response["output"])
        st.session_state.conversations.append(
            ({"role": "AI", "content": response["output"]}, 0)
        )
        st.session_state.conversations_text.append(AIMessage(response["output"]))
    return


def show_chat_history():
    for temp in st.session_state["conversations"]:
        conv = temp[0]
        fig = temp[1]
        role = conv["role"]
        if role == "System":
            continue
        with st.chat_message(role):
            st.markdown(conv["content"])
            if fig:
                st.pyplot(fig)
