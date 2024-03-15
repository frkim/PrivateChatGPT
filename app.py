import streamlit as st
import os
import openai

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from rich.console import Console
from langchain_community.utilities import BingSearchAPIWrapper

from langchain.agents import load_tools, initialize_agent

from langchain_core.tools import Tool

load_dotenv()


def get_azure_openai_client():
    client = AzureChatOpenAI(
        azure_endpoint = os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment = os.environ['AZURE_OPENAI_MODEL_NAME'],
        api_key = os.environ['AZURE_OPENAI_API_KEY'],
        api_version = "2023-09-01-preview",
        streaming=True
    )

    return client

config = """
AZURE_OPENAI_ENDPOINT={AZURE_OPENAI_ENDPOINT}
AZURE_OPENAI_MODEL_NAME={AZURE_OPENAI_MODEL_NAME}
AZURE_OPENAI_API_KEY={AZURE_OPENAI_API_KEY}
BING_SUBSCRIPTION_KEY={BING_SUBSCRIPTION_KEY}
"""
config = config.format(AZURE_OPENAI_ENDPOINT= os.getenv('AZURE_OPENAI_ENDPOINT'), 
                       AZURE_OPENAI_MODEL_NAME= os.getenv('AZURE_OPENAI_MODEL_NAME'),
                       AZURE_OPENAI_API_KEY= os.getenv('AZURE_OPENAI_API_KEY'),
                       BING_SUBSCRIPTION_KEY= os.getenv('BING_SUBSCRIPTION_KEY')
                       )

with st.expander("Parameters"):
    st.code(config)


st.title("🔎 LangChain - Chat with search")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

systemPrompt = """Vous êtes assistant(e) virtuel(le). Vous devez répondre aux questions des utilisateurs en utilisant les outils à votre disposition.
        Ne faites pas de suppositions sur les valeurs à introduire dans les fonctions. Demandez des éclaircissements si la demande d'un utilisateur est ambiguë.
 Ne mettez pas de valeurs vides dans les fonctions. Ne répondez pas à des questions sans rapport avec le sujet. Répondez en priorité en français sauf si la demande est formulée en anglais"""

if prompt := st.chat_input(placeholder="Type a question here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "system", "content": systemPrompt})
    st.chat_message("user").write(prompt)

    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    llm = get_azure_openai_client() #ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)

    # tools = load_tools(["google-serper"], llm, serper_api_key="83702167fca47fb43731ce846952fd3f368c7db6")
    tools = load_tools(["bing-search"], llm, bing_subscription_key=os.getenv('BING_SUBSCRIPTION_KEY'), bing_search_url="https://api.bing.microsoft.com/v7.0/search", search_kwargs={'mkt': 'fr-FR', 'setLang':'fr-FR', 'safeSearch': 'moderate'})

    #search = BingSearchAPIWrapper(k=1, bing_subscription_key="##################", bing_search_url="https://api.bing.microsoft.com/v7.0/search", search_kwargs={'mkt': 'fr-FR', 'setLang':'fr-FR', 'safeSearch': 'moderate'})

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, verbose=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "system", "content": systemPrompt})
        st.write(response)