import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pandas as pd
from llama_index.core import Document


PERSIST_DIR = "./storage"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192" 
CSV_FILE_PATH = "shl_assessments.csv"
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] or os.getenv("GROQ_API_KEY")


def load_data_from_csv(csv_path):
    """Loads assessment data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["Assessment Name", "URL", "Remote Testing Support",
                            "Adaptive/IRT Support", "Duration (min)", "Test Type"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: CSV file not found at {csv_path}")
    except ValueError as e:
        raise ValueError(f"Error reading CSV: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading CSV data: {e}")
    

def load_groq_llm():
    try:
        api_key = st.secrets.get["GROQ_API_KEY"] or os.getenv("GROQ_API_KEY")
    except KeyError:
        raise ValueError("GROQ_API_KEY not found in Streamlit secrets.")
    
    return Groq(model=LLM_MODEL, api_key=api_key, temperature=0.1)

def load_embeddings():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)    

def build_index(data):
    """Builds the vector index from the provided assessment data."""
    Settings.embed_model = load_embeddings()
    Settings.llm = load_groq_llm()

    documents = [Document(text=f"Name: {item['Assessment Name']}, URL: {item['URL']}, Remote Testing: {item['Remote Testing Support']}, Adaptive/IRT: {item['Adaptive/IRT Support']}, Duration: {item['Duration (min)']}, Type: {item['Test Type']}") for item in data]

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_chat_engine():
    """Loads the chat engine from the persisted index."""
    if not os.path.exists(PERSIST_DIR):
        return None

    Settings.embed_model = load_embeddings()
    Settings.llm = load_groq_llm()
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index.as_chat_engine(chat_mode="context", verbose=True)

def reset_index():
    """Resets the persisted index and chat history."""
    try:
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        st.success("Knowledge index reset successfully!")
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your SHL assessment assistant. How can I help you?"}]
        st.session_state["index_built"] = False
        if 'chat_engine' in st.session_state:
            del st.session_state['chat_engine']
        return None
    except Exception as e:
        st.error(f"Error resetting index: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="SHL Assessment Chatbot",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
    :root {
        --primary: #6eb5ff;
        --background: #000000;
        --card: #f0f2f6;
        --text: #1a1a1a;
        --background: #000000;
        --card: #f0f2f6;
        --text: #1a1a1a;
    }
    .stApp {
        background-color: var(--background) !important;
        color: var(--text) !important;
    }
    .stMarkdown, .stTextInput, .stChatMessage, .stChatInputContainer, .css-10trblm, .css-1cpxqw2 {
        color: var(--text) !important;
    }
    .stApp {
        background-color: var(--background) !important;
        color: var(--text) !important;
    }
    .stMarkdown, .stTextInput, .stChatMessage, .stChatInputContainer, .css-10trblm, .css-1cpxqw2 {
        color: var(--text) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    load_dotenv()
    os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
    os.environ["TORCH_DISABLE_STREAMLIT_WATCHER"] = "1"
    os.environ["LLAMA_INDEX_DISABLE_OPENAI"] = "1"

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "🤖 Hello! I'm your SHL assessment assistant. How can I help you?"
        }]
        st.session_state.messages = [{
            "role": "assistant",
            "content": "🤖 Hello! I'm your SHL assessment assistant. How can I help you?"
        }]
    if "index_built" not in st.session_state:
        st.session_state["index_built"] = False



    if not st.session_state["index_built"]:
        try:
            with st.spinner("Loading data and building index..."):
                assessment_data = load_data_from_csv(CSV_FILE_PATH)
                if assessment_data:
                    build_index(assessment_data)
                    st.session_state['chat_engine'] = load_chat_engine()
                    st.session_state["index_built"] = True
                else:
                    st.error("Failed to load assessment data. Please check the CSV file.")
        except Exception as e:
            st.error(f"Error initializing application: {e}")

    # --- Chat Interface ---
    chat_engine = st.session_state.get('chat_engine')
    if chat_engine:
        for msg in st.session_state.messages:
            icon = "🤖" if msg["role"] == "assistant" else "👤"
            icon = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"]):
                st.markdown(f"<span style='color: black;'>{icon} {msg['content']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: black;'>{icon} {msg['content']}</span>", unsafe_allow_html=True)

        if prompt := st.chat_input("Ask me about SHL assessments..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"<span style='color: black;'>👤 {prompt}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: black;'>👤 {prompt}</span>", unsafe_allow_html=True)

            with st.chat_message("assistant"):
                try:
                    # Add formatting instructions to the prompt
                    formatted_prompt = f"{prompt}. Please provide a list of all matching SHL assessments (minimum 1, maximum 10). For each assessment, include the following details: Assessment Name: [Name], URL: [URL], Remote Testing Support: [Yes/No], Adaptive/IRT Support: [Yes/No], Duration: [Duration], Test Type: [Type]. If there are no matching assessments, please state that."
                    response = chat_engine.chat(formatted_prompt)
                    st.markdown(f"<span style='color: black;'>🤖 {response.response}</span>", unsafe_allow_html=True)
                    # Add formatting instructions to the prompt
                    formatted_prompt = f"{prompt}. Please provide a list of all matching SHL assessments (minimum 1, maximum 10). For each assessment, include the following details: Assessment Name: [Name], URL: [URL], Remote Testing Support: [Yes/No], Adaptive/IRT Support: [Yes/No], Duration: [Duration], Test Type: [Type]. If there are no matching assessments, please state that."
                    response = chat_engine.chat(formatted_prompt)
                    st.markdown(f"<span style='color: black;'>🤖 {response.response}</span>", unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response.response})
                except Exception as e:
                    st.error(f"An error occurred during chat: {e}")

    else:
        st.info("💬 Chat is ready! Ask me anything about SHL assessments.")
        st.info("💬 Chat is ready! Ask me anything about SHL assessments.")

if __name__ == "__main__":
    main()

