import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

# Load environment variables (if any, for example, API keys, though not strictly needed for local Ollama)
load_dotenv()

# --- Configuration ---
VECTOR_DB_PATH = "faiss_index" # Directory to save the FAISS vector store
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OLLAMA_MODEL = "llama2:13b" # Make sure you have this model pulled in Ollama (e.g., ollama pull llama3)

# --- Helper Functions ---

def load_documents(uploaded_files):
    """Loads various document types using LangChain's loaders."""
    documents = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = os.path.join("temp", uploaded_file.name)

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt" or file_extension == ".md":
            loader = TextLoader(file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            # UnstructuredWordDocumentLoader requires 'unstructured' and its dependencies
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".pptx" or file_extension == ".ppt":
            # UnstructuredPowerPointLoader requires 'unstructured' and its dependencies
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
            continue
        documents.extend(loader.load())
    return documents

def get_text_chunks(documents):
    """Splits documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vector_store(text_chunks):
    """Creates or loads a FAISS vector store from text chunks."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    if os.path.exists(VECTOR_DB_PATH):
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        # Assuming you want to add new documents to an existing store
        # You might need to update this logic if you want to completely rebuild or re-index
        st.info("Existing vector store loaded. Appending new documents if any.")
        # For simplicity, we'll just create a new one if files are uploaded.
        # In a real app, you'd manage adding to/updating the existing store.
        if text_chunks: # Only add if new chunks are provided
            new_vectorstore = FAISS.from_documents(text_chunks, embeddings)
            vectorstore.merge_from(new_vectorstore)
            vectorstore.save_local(VECTOR_DB_PATH)
            st.success("New documents added to the vector store.")
    else:
        st.info("Creating new vector store...")
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        st.success("Vector store created and saved.")
    return vectorstore

def get_conversational_chain():
    """Defines the conversational chain with a rephrasing prompt."""
    prompt_template = """
    You are a helpful and knowledgeable human assistant. Your goal is to provide clear, concise, and natural-sounding answers based on the provided context. If the answer is not in the context, politely state that you don't have enough information. Avoid jargon and always respond as if you're having a conversation with another human.

    Context:
    {context}

    Question:
    {question}

    Your thoughtful and rephrased response:
    """
    
    # We use a slight variation for the rephrasing part to encourage natural language.
    # The conversational chain automatically handles context.
    
    llm = Ollama(model=OLLAMA_MODEL)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit UI ---

st.set_page_config(page_title="Conversational RAG with Ollama", layout="wide")

st.title("📚 Your Personal Knowledge Navigator")
st.markdown("Upload your documents (PDF, PPT, DOC, MD, TXT), and let's chat about them!")

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Upload Your Knowledge")
    uploaded_files = st.file_uploader(
        "Choose your files (PDF, PPT, DOC, MD, TXT)",
        type=["pdf", "ppt", "pptx", "doc", "docx", "md", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Documents"):
            os.makedirs("temp", exist_ok=True) # Create a temporary directory for uploaded files
            with st.spinner("Loading and processing documents... This might take a moment."):
                documents = load_documents(uploaded_files)
                if documents:
                    text_chunks = get_text_chunks(documents)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("Documents processed and knowledge base updated!")
                else:
                    st.error("No supported documents were loaded. Please check file types.")
            # Clean up temporary files
            for f in os.listdir("temp"):
                os.remove(os.path.join("temp", f))
            os.rmdir("temp")
    else:
        st.info("Upload documents to start building your knowledge base.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.info("No knowledge base loaded yet. Please upload and process documents in the sidebar.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vector_store:
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_store.similarity_search(prompt)
            chain = get_conversational_chain()
            
            # The LLM will rephrase the answer based on the prompt template.
            # We pass the question directly as the 'question' in the chain.
            response = chain.run(input_documents=docs, question=prompt)
            
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.chat_message("assistant").write("Please upload and process documents first to enable the knowledge base.")