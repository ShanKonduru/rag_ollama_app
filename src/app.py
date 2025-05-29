import streamlit as st
import os
from dotenv import load_dotenv
import shutil
import pandas as pd
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

BASE_VECTOR_DB_DIR = "knowledge_base"
MASTER_VERSION_CONTROL_FILE = os.path.join(
    BASE_VECTOR_DB_DIR, "MasterVersionControl.csv")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
OLLAMA_MODEL = "llama2:13b"

# --- Helper Functions ---


def ensure_base_directories_exist():
    """Ensures the base directory for vector stores and temp directory exist."""
    os.makedirs(BASE_VECTOR_DB_DIR, exist_ok=True)
    os.makedirs("temp", exist_ok=True)


def get_next_version_number():
    """Determines the next sequential version number (e.g., v1.0, v2.0)."""
    if not os.path.exists(MASTER_VERSION_CONTROL_FILE):
        return "v1.0"

    df = pd.read_csv(MASTER_VERSION_CONTROL_FILE)
    if df.empty:
        return "v1.0"

    latest_version = df['Version'].str.replace('v', '').astype(float).max()
    next_version = f"v{latest_version + 1:.1f}"
    return next_version


def update_master_version_control(version, file_details):
    """Updates the MasterVersionControl.csv with details of the new vector store version."""
    new_entry = {
        "Version": version,
        "Timestamp": datetime.now().isoformat(),
        "Files_Used": file_details
    }

    df_new = pd.DataFrame([new_entry])

    if not os.path.exists(MASTER_VERSION_CONTROL_FILE) or pd.read_csv(MASTER_VERSION_CONTROL_FILE).empty:
        df_new.to_csv(MASTER_VERSION_CONTROL_FILE, index=False)
    else:
        df_existing = pd.read_csv(MASTER_VERSION_CONTROL_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(MASTER_VERSION_CONTROL_FILE, index=False)


def load_documents(uploaded_files):
    """Loads various document types using LangChain's loaders."""
    documents = []
    uploaded_file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_paths.append(file_path)

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt" or file_extension == ".md":
            loader = TextLoader(file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            # Unstructured WordDocument Loader requires 'unstructured' and its dependencies
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".pptx" or file_extension == ".ppt":
            # Unstructured PowerPoint Loader requires 'unstructured' and its dependencies
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            continue
        documents.extend(loader.load())
    return documents, uploaded_file_paths


def get_text_chunks(documents):
    """Splits documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_and_save_vector_store(text_chunks, version):
    """Creates a new FAISS vector store and saves it with a version."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    version_dir = os.path.join(BASE_VECTOR_DB_DIR, version)
    os.makedirs(version_dir, exist_ok=True)

    st.info(f"Creating new vector store for version {version}...")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    vectorstore.save_local(folder_path=version_dir,
                           index_name="knowledge_base")
    st.success(f"Vector store {version} created and saved at {version_dir}.")
    return vectorstore


def load_vector_store_by_version(version):
    """Loads a specific FAISS vector store by its version."""
    version_dir = os.path.join(BASE_VECTOR_DB_DIR, version)
    if os.path.exists(version_dir):
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        st.info(f"Loading vector store version {version}...")
        vectorstore = FAISS.load_local(
            version_dir, embeddings, allow_dangerous_deserialization=True, index_name="knowledge_base")
        st.success(f"Vector store version {version} loaded successfully!")
        return vectorstore
    else:
        st.error(
            f"Vector store for version {version} not found at {version_dir}.")
        return None


def get_conversational_chain():
    """Defines the conversational chain with a rephrasing prompt."""
    prompt_template = """
    You are a helpful and knowledgeable human assistant. 
    Your goal is to provide clear, concise, and natural-sounding answers based on the provided context. 
    If the answer is not in the context, politely state that you don't have enough information. 
    Avoid jargon and always respond as if you're having a conversation with another human.

    Context:
    {context}

    Question:
    {question}

    Your thoughtful and rephrased response:
    """

    llm = Ollama(model=OLLAMA_MODEL)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain


st.set_page_config(page_title="Conversational RAG with Ollama", layout="wide")

st.title("📚 Shan Konduru - Knowledge Navigator")
st.markdown("Upload your documents (PDF, PPT, DOC, MD, TXT), or load a saved knowledge base, and let's chat about them!")

ensure_base_directories_exist()

with st.sidebar:
    st.header("Build or Load Your Knowledge Base")

    st.subheader("1. Build a New Knowledge Base")
    uploaded_files = st.file_uploader(
        "Choose new files (PDF, PPT, DOC, MD, TXT)",
        type=["pdf", "ppt", "pptx", "doc", "docx", "md", "txt"],
        accept_multiple_files=True,
        key="new_files_uploader"
    )

    if uploaded_files:
        if st.button("Create New Knowledge Base"):
            with st.spinner("Loading and processing documents... This might take a moment."):
                documents, uploaded_file_paths = load_documents(uploaded_files)
                if documents:
                    text_chunks = get_text_chunks(documents)
                    current_version = get_next_version_number()
                    st.session_state.vector_store = create_and_save_vector_store(
                        text_chunks, current_version)

                    file_names = ", ".join(
                        [os.path.basename(p) for p in uploaded_file_paths])
                    file_paths_str = ", ".join(uploaded_file_paths)
                    update_master_version_control(
                        current_version, f"Names: {file_names} | Paths: {file_paths_str}")

                    st.session_state.current_kb_version = current_version
                    st.success(
                        f"New knowledge base '{current_version}' created and ready!")
                else:
                    st.error(
                        "No supported documents were loaded. Please check file types.")
            if os.path.exists("temp"):
                shutil.rmtree("temp")
            os.makedirs("temp", exist_ok=True)

    st.markdown("---")

    st.subheader("2. Load an Existing Knowledge Base")

    available_versions = []
    if os.path.exists(MASTER_VERSION_CONTROL_FILE):
        try:
            df_versions = pd.read_csv(MASTER_VERSION_CONTROL_FILE)
            if not df_versions.empty:
                available_versions = df_versions['Version'].tolist()
                available_versions.sort(key=lambda x: float(
                    x.replace('v', '')), reverse=True)
        except pd.errors.EmptyDataError:
            st.info(
                "MasterVersionControl.csv is empty. No previous versions to load.")

    if available_versions:
        selected_version = st.selectbox(
            "Select a saved knowledge base version:",
            options=["-- Select --"] + available_versions,
            key="version_selector"
        )

        if selected_version != "-- Select --":
            if st.button(f"Load '{selected_version}' Knowledge Base"):
                st.session_state.vector_store = load_vector_store_by_version(
                    selected_version)
                if st.session_state.vector_store:
                    st.session_state.current_kb_version = selected_version
                    st.success(f"Knowledge base '{selected_version}' loaded!")
                else:
                    st.error(
                        f"Failed to load knowledge base '{selected_version}'.")
    else:
        st.info("No saved knowledge bases found. Create one first!")

    st.markdown("---")
    if "current_kb_version" in st.session_state and st.session_state.current_kb_version:
        st.info(
            f"**Active Knowledge Base:** {st.session_state.current_kb_version}")
    else:
        st.info("No knowledge base currently active.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_kb_version" not in st.session_state:
    st.session_state.current_kb_version = None

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

if prompt := st.chat_input("Ask me anything about the active knowledge base..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vector_store:
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_store.similarity_search(prompt)
            chain = get_conversational_chain()

            response = chain.run(input_documents=docs, question=prompt)

            st.chat_message("assistant").write(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})
    else:
        st.chat_message("assistant").write(
            "Please load or create a knowledge base first to start chatting.")
