from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain_ollama import OllamaLLM


class RagFile:

    def __init__(self, file_path: str, persist_directory: str = "db"):

        self.embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.qa_llm = OllamaLLM(model="deepseek-r1:14b")

        self.file_path = file_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model)

        loader = TextLoader(self.file_path)
        documents = loader.load()
        self.vectorstore = Chroma.from_documents(documents, 
                                                 self.embeddings, 
                                                 persist_directory=self.persist_directory)

        self.qa_chain = RetrievalQA.from_chain_type(llm=self.qa_llm,
                                                    chain_type="stuff",
                                                    retriever=self.vectorstore.as_retriever(),
                                                    #return_source_documents=True,  # tool expects a simple response only.
                                                    )

    def search(self, question: str) -> str:
        """
        Tool to query the RAG system with a question.
        """
        return self.qa_chain.run(question)  # use invoke instead of run() if needed

    def create_tool(self) -> Tool:
        """Create a tool for querying the RAG system."""
        return Tool(
            name="RAG File Tool",
            #func=self.search,
            func=lambda question: self.qa_chain.run(question),  # maybe use invoke instead of run()
            description="Tool to query the RAG system with a question."
        )