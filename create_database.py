#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load PDF
loader = PyPDFLoader(r"D:\yt_shreyansh\gen_ai\rag_project_lec2\document_loader\deeplearning.pdf")
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Embedding model
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)