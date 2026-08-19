import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(
    page_title="DocuMind — AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind — AI Document Assistant")
st.write("Upload a PDF and ask questions about it.")


# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload your book / PDF",
    type=["pdf"]
)


if uploaded_file:

    # Save uploaded PDF temporarily
    pdf_path = "uploaded_book.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")


    # -------------------------------
    # Create embeddings
    # -------------------------------

    embedding_model = MistralAIEmbeddings(
        model="mistral-embed"
    )


    # -------------------------------
    # Process PDF
    # -------------------------------

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)


            # -------------------------------
            # Store in ChromaDB
            # -------------------------------

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )

            st.session_state["vectorstore"] = vectorstore

        st.success("✅ Document processed successfully!")


# -------------------------------
# Question Answering
# -------------------------------

if "vectorstore" in st.session_state:

    st.divider()

    st.subheader("💬 Ask questions about your document")

    query = st.text_input(
        "Enter your question"
    )


    if query:

        vectorstore = st.session_state["vectorstore"]

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )


        # Retrieve documents

        docs = retriever.invoke(query)


        # Create context

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )


        # -------------------------------
        # Mistral LLM
        # -------------------------------

        llm = ChatMistralAI(
            model="mistral-small-2506"
        )


        # -------------------------------
        # Prompt
        # -------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
                ),
                (
                    "human",
                    """Context:

{context}

Question:

{question}
"""
                )
            ]
        )


        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": query
            }
        )


        # Generate response

        with st.spinner("Thinking..."):

            response = llm.invoke(final_prompt)


        st.subheader("🤖 AI Answer")

        st.write(response.content)
