# from langchain_community.document_loaders import PyPDFLoader
# loader=PyPDFLoader("document_loader/gru.pdf")
# docs=loader.load()
# print(docs[13])

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document loaders/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)