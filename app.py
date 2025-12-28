import os
import tempfile
import streamlit as st

# -------- LangChain Core --------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------- LLM --------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# -------- Document Loaders --------
from langchain_community.document_loaders import PyPDFLoader

# -------- Text Splitters --------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------- Vector Stores --------
from langchain_community.vectorstores import Chroma

# -------- Chains --------
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="PDF RAG App", layout="wide")
st.title("📄 PDF RAG Application (LangChain + Streamlit)")

# API Key
openai_api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password"
)

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

query = st.text_input("Ask a question from the document")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

if uploaded_file and query and openai_api_key:

    with st.spinner("Processing document..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # 1️⃣ Load Document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 2️⃣ Split Text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)

        # 3️⃣ Embeddings
        embeddings = OpenAIEmbeddings()

        # 4️⃣ Vector Store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        # 5️⃣ Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # 6️⃣ LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # 7️⃣ Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based only on the provided context.

            Context:
            {context}

            Question:
            {input}
            """
        )

        # 8️⃣ Document Chain
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        # 9️⃣ Retrieval Chain (RAG)
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        # 🔟 Invoke Chain
        response = retrieval_chain.invoke({"input": query})

        st.success("Answer:")
        st.write(response["answer"])
