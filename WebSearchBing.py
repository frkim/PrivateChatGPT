from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType, create_react_agent, AgentExecutor

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

    
load_dotenv()

def show_result(result):
    console = Console()
    console.print(Markdown(f"##### Question : \n{result['input']}"))
    console.print(f"##### Answer : \n {result['output']}")


llm = AzureChatOpenAI(azure_deployment="gpt-4-turbo", api_version="2023-09-01-preview")

tools = load_tools(["bing-search"], llm=llm)
question = "Who won 2024 super bowl? What was score and who was MVP?"

agent= initialize_agent(tools,llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,verbose=False)
response = agent.invoke (question)

show_result(response)

prompt_template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(prompt_template)
tools = load_tools(["bing-search"], llm=llm)
agent1 = create_react_agent(llm, tools,prompt)
agent_executor = AgentExecutor(agent=agent1, tools=tools,verbose=True)
response = agent_executor.invoke({"input": question})

show_result(response)

site_question = "(site:https://www.fda.gov/food) Whar is sampling? Explain process, purpose and how it is used in food safety?"
response = agent_executor.invoke({"input": site_question})

show_result(response)


