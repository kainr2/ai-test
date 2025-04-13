from langchain.chains import RetrievalQA
#from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
#from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

import os


file_path = "resources/manual.txt"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")
loader = TextLoader(file_path)
docs = loader.load()

## embedding model
# embedding_model = OpenAiEmbeddings()
# embedding_model = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1")

texts = [doc.page_content for doc in docs]  # Extract text from Document objects
embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(texts, convert_to_tensor=True)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

## save and read from vector store
db = Chroma.from_documents(docs, embedding_model)
retriever = db.as_retriever()

## LLM 
try:
    #llm = ChatOpenAI(model_name="gpt-4")
    llm = OllamaLLM(model="llama3.2:latest")
except Exception as e:
    raise RuntimeError(f"Failed to initialize the LLM: {e}")


qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)


# Loop to continuously get user input
while True:
    # Get user input for the query
    query = input("Please enter your query (or type 'exit' to quit): ")
    if query.lower() in ['exit', 'quit']:
        print("Exiting the program. Goodbye!")
        break
    # Run the query through the RetrievalQA chain
    #response = qa.run(query)
    response = qa.invoke({"query": query})
    print("...")
    print("Answer:", response["result"])
    print("--------------------\n")
