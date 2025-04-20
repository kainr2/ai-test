from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory


# Load config
load_dotenv()


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.agents import Tool

# Load and index documents
loader = TextLoader("resources/manual.txt")
docs = loader.load()
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()
rag_chain = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=OllamaLLM(model="deepseek-r1:14b"),
    #return_source_documents=True
)

tools = [
    Tool(
        name="RAG Retriever",
        func=lambda q: rag_chain.run(q),
        description="Useful for questions that can be answered from internal documentation."
    ),
]


## LLM 
try:
    llm = OllamaLLM(model="llama3.2:latest")
    #llm = OllamaLLM(model="deepseek-r1:14b")
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

# Run the query
query = "What is X200 firmware version?"
response = agent.invoke({"input": query})
print("...")
print(f"Answer: {response}")
print("...")
print(f"Answer: {response['output']}")
print("--------------------\n")
