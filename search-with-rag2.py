import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory

from tools.google_search import GoogleSearch
from tools.rag_file import RagFile



# Load config
load_dotenv()

# List of tools
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
file_path = "resources/manual.txt"   

tool_rag_file = RagFile(file_path)
tools = [
    GoogleSearch(GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID).create_tool(),
    tool_rag_file.create_tool(),
]


## LLM 
try:
    llm = OllamaLLM(model="llama3.2:latest")
except Exception as e:
    raise RuntimeError(f"Failed to initialize the LLM: {e}")


# Chat Memory for context
memory = ConversationBufferMemory(memory_key="chat_history", 
                                  return_messages=True,
                                  max_memory=5)

# Agents
agent = initialize_agent(tools=tools, 
                         llm=llm, 
                         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                         memory=memory,
                         handle_parsing_errors=True,
                         max_iterations=3,
                         verbose=True)

# Loop to continuously get user input
while True:
    # Get user input for the query
    query = input("Please enter your query (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        print("Exiting the program. Goodbye!")
        break
    response = agent.invoke({"input": query})
    print("...")
    print(f"Answer: {response}")
    print("...")
    print(f"Answer: {response['output']}")
    print("--------------------\n")
