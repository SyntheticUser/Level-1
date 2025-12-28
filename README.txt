PDF RAG Application using LangChain & Streamlit
================================================

Project Overview
----------------
This project implements a Retrieval-Augmented Generation (RAG) application using
LangChain v0.2+ and Streamlit. Users can upload PDF documents and ask natural
language questions, which are answered based on retrieved document context.

Architecture
------------
PDF Upload
 → Document Loader (PyPDFLoader)
 → Text Splitter (RecursiveCharacterTextSplitter)
 → Embeddings (OpenAIEmbeddings)
 → Vector Store (ChromaDB)
 → Retriever
 → RAG Chain
 → Streamlit UI Response

Tech Stack
----------
Python           : 3.11 / 3.12
Framework        : LangChain v0.2+
UI               : Streamlit
LLM              : OpenAI
Embeddings       : OpenAI Embeddings
Vector Database  : ChromaDB
PDF Parsing      : PyPDF

Project Structure
-----------------
project-root/
 ├── app.py
 ├── requirements.txt
 └── README.txt

Installation (Recommended)
--------------------------
conda create -n rag python=3.11 -y
conda activate rag
pip install -r requirements.txt

Installation (Anaconda Base)
----------------------------
/opt/anaconda3/bin/python -m pip install -r requirements.txt

Running the App
---------------
streamlit run app.py

Usage
-----
1. Enter OpenAI API key in the sidebar
2. Upload a PDF file
3. Ask questions related to the document

Environment Variables
---------------------
export OPENAI_API_KEY="your-api-key"

Important Dependency Notes
--------------------------
LangChain v0.2+ requires modular packages:
- langchain
- langchain-core
- langchain-community
- langchain-openai
- langchain-text-splitters

Optional dependencies used:
- pypdf (PDF loading)
- chromadb (vector database)

Protobuf Fix (If Needed)
------------------------
pip install "protobuf<6,>=3.20"
Recommended: protobuf==4.25.3

Sanity Check
------------
python - <<EOF
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import chromadb, pypdf
print("RAG environment is ready")
EOF

Future Enhancements
-------------------
- Persistent vector store
- Conversational RAG
- Multi-document ingestion
- Local embeddings
- Agentic RAG with LangGraph

License
-------
For educational and demonstration purposes.
