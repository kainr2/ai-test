import requests
import os

from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory

from tools.google_search import GoogleSearch


# Load config
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")


# Search List:
# * <https://brave.com/search/api/>
# * <https://tavily.com/#pricing>
#
google_tool = GoogleSearch(GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID).create_tool()

## LLM 
try:
    llm = OllamaLLM(model="llama3.2:latest")
except Exception as e:
    raise RuntimeError(f"Failed to initialize the LLM: {e}")

# Search tools
tools = [google_tool]
# Chat Memory for context
memory = ConversationBufferMemory(memory_key="chat_history", 
                                  return_messages=True,
                                  max_memory=5)

# Agents
# single call, no memory
#agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# keep conversation history
agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                         memory=memory,
                         verbose=True)

# # Example query
# response = agent.run("Who is the current president of the United States as of April 2025?")
# print(response)

# Loop to continuously get user input
while True:
    # Get user input for the query
    query = input("Please enter your query (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        print("Exiting the program. Goodbye!")
        break
    # Run the query
    response = agent.invoke({"input": query})

    print("...")
    print(f"Answer: {response}")
    print("...")
    print(f"Answer: {response['output']}")
    print("--------------------\n")
