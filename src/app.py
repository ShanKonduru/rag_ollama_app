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

# Load environment variables (if any, for example, API keys, though not strictly needed for local Ollama)
load_dotenv()

# --- Configuration ---
BASE_VECTOR_DB_DIR = "knowledge_base" # Base directory for all versioned vector stores
MASTER_VERSION_CONTROL_FILE = os.path.join(BASE_VECTOR_DB_DIR, "MasterVersionControl.csv")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
OLLAMA_MODEL = "llama2:13b" # Make sure you have this model pulled in Ollama (e.g., ollama pull llama3)

# --- Helper Functions ---

def ensure_base_directories_exist():
    """Ensures the base directory for vector stores and temp directory exist."""
    os.makedirs(BASE_VECTOR_DB_DIR, exist_ok=True)
    os.makedirs("temp", exist_ok=True) # Ensure temp directory for uploads

def get_next_version_number():
    """Determines the next sequential version number (e.g., v1.0, v2.0)."""
    if not os.path.exists(MASTER_VERSION_CONTROL_FILE):
        return "v1.0"
    
    df = pd.read_csv(MASTER_VERSION_CONTROL_FILE)
    if df.empty:
        return "v1.0"
    
    # Extract version numbers, convert to float, find max, increment
    # Assumes versions are like 'vX.Y' and we only increment X for new builds
    latest_version = df['Version'].str.replace('v', '').astype(float).max()
    next_version = f"v{latest_version + 1:.1f}"
    return next_version

def get_knowledge_base_name(documents):
    """
    Uses LLM to suggest a descriptive name for the knowledge base
    based on the content of the first few documents.
    """
    llm = Ollama(model=OLLAMA_MODEL)

    # Take content from the first few documents for a concise overview
    # Limit to avoid sending too much text to LLM
    combined_content = ""
    for doc in documents[:5]: # Consider content from up to 5 documents
        combined_content += doc.page_content[:1000] # Take first 1000 chars from each
        if len(combined_content) > 3000: # Max 3000 chars for prompt
            break

    if not combined_content:
        return "Unnamed_Knowledge_Base"

    name_prompt = PromptTemplate(
        template="""Based on the following document content, suggest a short (2-5 words), descriptive, and suitable name for a knowledge base that contains this information. The name should be concise, use underscores instead of spaces, and contain only lowercase letters and numbers. Avoid file extensions or specific document names.

Content:
{content}

Suggested Name:""",
        input_variables=["content"]
    )

    try:
        response = llm.invoke(name_prompt.format(content=combined_content))
        # Basic cleaning: lowercase, replace spaces/dashes with underscores, remove special chars
        cleaned_name = "".join(char if char.isalnum() or char == '_' else '_' for char in response).lower()
        cleaned_name = "_".join(filter(None, cleaned_name.split('_'))) # Remove multiple underscores

        # Ensure it's not empty and has a reasonable length
        if not cleaned_name or len(cleaned_name) < 3:
            return "general_knowledge"

        # Limit length to avoid extremely long directory names
        return cleaned_name[:50]
    except Exception as e:
        st.warning(f"Could not generate a smart name for the knowledge base. Using default. Error: {e}")
        return "default_knowledge_base"

def update_master_version_control(version_label, file_details, generated_name): # <-- CHANGED
    """Updates the MasterVersionControl.csv with details of the new vector store version."""
    new_entry = {
        "Version": version_label,
        "Generated_Name": generated_name, # <-- NEW LINE
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
        # Save to temp directory
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_paths.append(file_path) # Store path for logging

        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

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

def create_and_save_vector_store(text_chunks, version_label, generated_name): # <-- CHANGED
    """Creates a new FAISS vector store and saves it with a version and generated name."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Combine version and generated name for a descriptive directory name
    version_dir_name = f"{version_label}_{generated_name}" # <-- CHANGED
    version_dir_path = os.path.join(BASE_VECTOR_DB_DIR, version_dir_name) # <-- CHANGED
    os.makedirs(version_dir_path, exist_ok=True)

    st.info(f"Creating new vector store '{version_label}' ({generated_name})...") # <-- CHANGED
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    vectorstore.save_local(version_dir_path) # <-- CHANGED
    st.success(f"Vector store '{version_label}' ({generated_name}) created and saved.") # <-- CHANGED
    return vectorstore

def load_vector_store_by_version_and_name(version_label, generated_name): # <-- NEW NAME & PARAMS
    """Loads a specific FAISS vector store by its version and generated name."""
    version_dir_name = f"{version_label}_{generated_name}" # <-- CHANGED
    version_dir_path = os.path.join(BASE_VECTOR_DB_DIR, version_dir_name) # <-- CHANGED

    if os.path.exists(version_dir_path):
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        st.info(f"Loading knowledge base '{version_label}' ({generated_name})...") # <-- CHANGED
        vectorstore = FAISS.load_local(version_dir_path, embeddings, allow_dangerous_deserialization=True) # <-- CHANGED
        st.success(f"Knowledge base '{version_label}' ({generated_name}) loaded successfully!") # <-- CHANGED
        return vectorstore
    else:
        st.error(f"Knowledge base for '{version_label}' ({generated_name}) not found at {version_dir_path}.") # <-- CHANGED
        return None

def load_vector_store_by_version(version):
    """Loads a specific FAISS vector store by its version."""
    version_dir = os.path.join(BASE_VECTOR_DB_DIR, version)
    if os.path.exists(version_dir):
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        st.info(f"Loading vector store version {version}...")
        vectorstore = FAISS.load_local(version_dir, embeddings, allow_dangerous_deserialization=True)
        st.success(f"Vector store version {version} loaded successfully!")
        return vectorstore
    else:
        st.error(f"Vector store for version {version} not found at {version_dir}.")
        return None

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
    
    llm = Ollama(model=OLLAMA_MODEL)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit UI ---

st.set_page_config(page_title="Conversational RAG with Ollama", layout="wide")

st.title("📚 Your Personal Knowledge Navigator")
st.markdown("Upload your documents (PDF, PPT, DOC, MD, TXT), or load a saved knowledge base, and let's chat about them!")

# Ensure base directories exist on app start
ensure_base_directories_exist()

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Build or Load Your Knowledge Base")

    # --- Upload and Process New Documents ---
    st.subheader("1. Build a New Knowledge Base")
    uploaded_files = st.file_uploader(
        "Choose new files (PDF, PPT, DOC, MD, TXT)",
        type=["pdf", "ppt", "pptx", "doc", "docx", "md", "txt"],
        accept_multiple_files=True,
        key="new_files_uploader"
    )

    if uploaded_files:
        if st.button("Create New Knowledge Base"):
            with st.spinner("Loading and processing documents and generating name..."): # <-- CHANGED SPINNER MESSAGE
                documents, uploaded_file_paths = load_documents(uploaded_files)
                if documents:
                    # --- NEW STEP: Get LLM-generated name ---
                    generated_name = get_knowledge_base_name(documents) # <-- NEW LINE
                    st.info(f"Suggested Knowledge Base Name: **{generated_name}**") # <-- NEW LINE

                    text_chunks = get_text_chunks(documents)
                    current_version_label = get_next_version_number() # <-- CHANGED VARIABLE NAME

                    st.session_state.vector_store = create_and_save_vector_store( # <-- CHANGED FUNCTION CALL
                        text_chunks, current_version_label, generated_name # <-- ADDED generated_name
                    )

                    # Log file details for this version, including the generated name
                    file_names = ", ".join([os.path.basename(p) for p in uploaded_file_paths])
                    file_paths_str = ", ".join(uploaded_file_paths)
                    update_master_version_control(current_version_label, f"Names: {file_names} | Paths: {file_paths_str}", generated_name) # <-- ADDED generated_name

                    st.session_state.current_kb_version_label = current_version_label # <-- CHANGED SESSION STATE KEY
                    st.session_state.current_kb_generated_name = generated_name # <-- NEW SESSION STATE KEY
                    st.success(f"New knowledge base '{current_version_label}' ({generated_name}) created and ready!") # <-- CHANGED SUCCESS MESSAGE
                else:
                    st.error("No supported documents were loaded. Please check file types.")
            # Clean up temporary files
            if os.path.exists("temp"):
                shutil.rmtree("temp") # Remove temp directory and its contents
            os.makedirs("temp", exist_ok=True) # Recreate empty temp directory for next uploads
    
    st.markdown("---")

    # --- Load Existing Knowledge Base ---
    st.subheader("2. Load an Existing Knowledge Base")
    
    # Get available versions from MasterVersionControl.csv
    available_versions_data = [] # Store tuples of (version_label, generated_name) # <-- CHANGED
    if os.path.exists(MASTER_VERSION_CONTROL_FILE):
        try:
            df_versions = pd.read_csv(MASTER_VERSION_CONTROL_FILE)
            if not df_versions.empty and 'Generated_Name' in df_versions.columns: # <-- CHANGED CONDITION
                # Create a display string for the selectbox: "vX.Y (generated_name)"
                available_versions_data = [(row['Version'], row['Generated_Name']) for index, row in df_versions.iterrows()] # <-- CHANGED
                # Sort numerically by version label
                available_versions_data.sort(key=lambda x: float(x[0].replace('v', '')), reverse=True) # <-- CHANGED
            elif not df_versions.empty: # Handle older CSVs without Generated_Name column # <-- NEW BLOCK
                st.warning("MasterVersionControl.csv found but lacks 'Generated_Name' column. Please recreate if naming is desired.")
                available_versions_data = [(row['Version'], "Unnamed_KB") for index, row in df_versions.iterrows()]
                available_versions_data.sort(key=lambda x: float(x[0].replace('v', '')), reverse=True)

        except pd.errors.EmptyDataError:
            st.info("MasterVersionControl.csv is empty. No previous versions to load.")

    display_options = ["-- Select --"] + [f"{v_label} ({g_name})" for v_label, g_name in available_versions_data] # <-- CHANGED DISPLAY OPTIONS

    selected_display_option = st.selectbox( # <-- CHANGED VARIABLE NAME
        "Select a saved knowledge base version:",
        options=display_options, # <-- CHANGED
        key="version_selector"
    )

    selected_version_label = None # <-- NEW VARS
    selected_generated_name = None # <-- NEW VARS
    if selected_display_option != "-- Select --":
        # Extract version label and generated name from the selected string
        try:
            # Assumes format "vX.Y (name_with_underscores)"
            parts = selected_display_option.split(' ', 1) # Split at first space
            selected_version_label = parts[0]
            selected_generated_name = parts[1][1:-1] # Remove parentheses
        except IndexError:
            st.error("Could not parse selected version name. Please re-check MasterVersionControl.csv format.")
            selected_version_label = None
            selected_generated_name = None

    if selected_version_label and selected_generated_name: # <-- CHANGED CONDITION
        if st.button(f"Load '{selected_version_label}' ({selected_generated_name}) Knowledge Base"): # <-- CHANGED BUTTON TEXT
            st.session_state.vector_store = load_vector_store_by_version_and_name( # <-- CHANGED FUNCTION CALL
                selected_version_label, selected_generated_name # <-- CHANGED PARAMS
            )
            if st.session_state.vector_store:
                st.session_state.current_kb_version_label = selected_version_label # <-- CHANGED SESSION STATE KEYS
                st.session_state.current_kb_generated_name = selected_generated_name # <-- NEW SESSION STATE KEY
                st.success(f"Knowledge base '{selected_version_label}' ({selected_generated_name}) loaded!") # <-- CHANGED SUCCESS MESSAGE
            else:
                st.error(f"Failed to load knowledge base '{selected_version_label}' ({selected_generated_name}).") # <-- CHANGED ERROR MESSAGE
    else: # <-- CHANGED ELSE BLOCK
        st.info("No saved knowledge bases found or selected.")

    st.markdown("---")
    if "current_kb_version_label" in st.session_state and st.session_state.current_kb_version_label: # <-- CHANGED SESSION STATE KEY
        st.info(f"**Active Knowledge Base:** {st.session_state.current_kb_version_label} ({st.session_state.current_kb_generated_name})") # <-- CHANGED DISPLAY
    else:
        st.info("No knowledge base currently active.")
        
    st.markdown("---")
    if "current_kb_version" in st.session_state and st.session_state.current_kb_version:
        st.info(f"**Active Knowledge Base:** {st.session_state.current_kb_version}")
    else:
        st.info("No knowledge base currently active.")


# Initialize chat history and vector store in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_kb_version" not in st.session_state:
    st.session_state.current_kb_version = None

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the active knowledge base..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.vector_store:
        with st.spinner("Thinking..."):
            docs = st.session_state.vector_store.similarity_search(prompt)
            chain = get_conversational_chain()
            
            response = chain.run(input_documents=docs, question=prompt)
            
            st.chat_message("assistant").write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.chat_message("assistant").write("Please load or create a knowledge base first to start chatting.")