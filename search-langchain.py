import os
import requests

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama


from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseLLM

from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import MessageGraph
from langgraph.graph.state import CompiledStateGraph
from langchain.schema import HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage

# local tools
from tools.google_search import GoogleSearch
from tools.rag_file import RagFile
from langchain_core.tools import tool

# Load config
load_dotenv()

# Create tools
def init_tools():
    """Create and return a list of tools for the agent."""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    google_search = GoogleSearch(GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID)
    file_path = "resources/manual.txt"
    rag_file = RagFile(file_path)
    return {
        "google_search": google_search,
        "rag_file": rag_file,
    }
preset_tools = init_tools()


@tool
def celsius_to_fahrenheit(temp_c: float) -> float:
    """
    Convert Celsius to Fahrenheit.
    Args:
        temp_c: temperature
    """
    return (temp_c * 9/5) + 32

@tool
def google(query: str) -> str:
    """
    Search the web using Google API and return the abstract text or description.
    """
    return preset_tools["google_search"].search(query)

@tool
def rag_file(query: str) -> str:
    """
    Tool to query the RAG system with a question.
    """
    return preset_tools["rag_file"].search(query)

def create_tools() -> list:    
    #tools = [ google_search.search ]
    tools = [celsius_to_fahrenheit, 
             google, 
             rag_file]
    return tools


def create_llm() -> BaseLLM:
    """Create and return an instance of the LLM."""
    # Use chat model for react agent.
    # https://medium.com/@diwakarkumar_18755/hands-on-guide-to-react-agents-using-langgraph-and-ollama-9e9897e9695c
    try:
        #return OllamaLLM(model="llama3.2:latest")
        #return OllamaLLM(model="deepseek-r1:14b")
        return ChatOllama(model="llama3.2:latest")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the LLM: {e}")


sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search")
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Create agent for LangGraph
def create_agent(tools: dict) -> CompiledStateGraph:
    """Create a LangGraph agent with the specified LLM and tools."""

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)    # Assistant node
    builder.add_node("tools", ToolNode(tools))  # Tools node
    builder.add_edge(START, "assistant")    # Connect start to assistant
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")  # Connect tools back to assistant
    graph = builder.compile()
    return graph

state = []
tools = create_tools()
llm = create_llm()
llm_with_tools = llm.bind_tools(tools)  # Bind tools to the LLM
agent_executor = create_agent(tools)

def print_after_done(agent_executor, messages):
    result = agent_executor.invoke({"messages": messages})
    for msg in result["messages"]:
        msg.pretty_print()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, FunctionMessage
def print_each_step(agent_executor, messages):
    print("Starting agent execution...")
    counter = 0
    for step in agent_executor.stream({"messages": messages}):
        print("-----------------------------------------")
        print("count: ", counter)
        counter += 1
        key = next(iter(step))  # Could be any node name: 'assistant' or 'tools'
        messages = step.get(key, {}).get("messages", [])
        # print only the new message(s)
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, AIMessage):
                print("🤖 AI:", last_msg.content)
            elif isinstance(last_msg, ToolMessage):
                print("🔧 Tool:", last_msg.content)
            elif isinstance(last_msg, FunctionMessage):
                print("🧠 Function:", last_msg.content)
            elif isinstance(last_msg, HumanMessage):
                print("👤 You:", last_msg.content)        


# Query and print the result
#user_input = input("Enter your query: ")
user_input = "where can i download the openai codex cli?"
messages = [HumanMessage(content=user_input),]
print_each_step(agent_executor, messages)
#print_after_done(agent_executor, messages)
