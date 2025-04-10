from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pandas as pd
from llama_index.core import Document

app = Flask(__name__)

# --- Configuration ---
PERSIST_DIR = "./storage"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"
CSV_FILE_PATH = "shl_assessments.csv"

# --- Root Route (for health check) ---
@app.route("/", methods=["GET"])
def home():
    return "🧠 SHL Chatbot API is running!", 200

# --- Utility Functions ---
def load_groq_llm():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file or environment variables")
    return Groq(model=LLM_MODEL, api_key=api_key, temperature=0.1)

def load_embeddings():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)

def load_data_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["Assessment Name", "URL", "Remote Testing Support",
                            "Adaptive/IRT Support", "Duration (min)", "Test Type"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

def build_index(data):
    Settings.embed_model = load_embeddings()
    Settings.llm = load_groq_llm()
    documents = [
        Document(text=f"Name: {item['Assessment Name']}, URL: {item['URL']}, Remote Testing: {item['Remote Testing Support']}, Adaptive/IRT: {item['Adaptive/IRT Support']}, Duration: {item['Duration (min)']}, Type: {item['Test Type']}")
        for item in data
    ]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_chat_engine():
    if not os.path.exists(PERSIST_DIR):
        return None
    Settings.embed_model = load_embeddings()
    Settings.llm = load_groq_llm()
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index.as_chat_engine(chat_mode="context", verbose=True)

# --- Load or Build Index ---
try:
    chat_engine = load_chat_engine()
    if chat_engine is None:
        assessment_data = load_data_from_csv(CSV_FILE_PATH)
        build_index(assessment_data)
        chat_engine = load_chat_engine()
except Exception as e:
    print(f"❌ Error initializing chat engine: {e}")
    chat_engine = None

# --- Endpoint ---
@app.route("/assessments", methods=["POST"])
def get_assessments():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if chat_engine:
        try:
            response = chat_engine.chat(query)
            results = []

            for node in response.source_nodes:
                try:
                    parts = node.node.text.split(", ")
                    results.append({
                        "assessment_name": parts[0].split(": ")[1] if len(parts) > 0 else "N/A",
                        "assessment_url": parts[1].split(": ")[1] if len(parts) > 1 else "N/A",
                        "remote_testing_support": parts[2].split(": ")[1] if len(parts) > 2 else "N/A",
                        "adaptive_irt_support": parts[3].split(": ")[1] if len(parts) > 3 else "N/A",
                        "duration": parts[4].split(": ")[1] if len(parts) > 4 else "N/A",
                        "test_type": parts[5].split(": ")[1] if len(parts) > 5 else "N/A"
                    })
                except:
                    results.append({"error": "Error parsing assessment info"})

            return jsonify({"query": query, "response": results}), 200

        except Exception as e:
            return jsonify({"error": f"Chat processing error: {e}"}), 500
    else:
        return jsonify({"error": "Chat engine not initialized"}), 500

# --- Entry Point for Local Debugging ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
import pandas as pd
from llama_index.core import Document

app = Flask(__name__)

# --- Configuration ---
PERSIST_DIR = "./storage"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192"
CSV_FILE_PATH = "shl_assessments.csv"

# --- Root Route (for health check) ---
@app.route("/", methods=["GET"])
def home():
    return "🧠 SHL Chatbot API is running!", 200

# --- Utility Functions ---
def load_groq_llm():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file or environment variables")
    return Groq(model=LLM_MODEL, api_key=api_key, temperature=0.1)

def load_embeddings():
    return HuggingFaceEmbedding(model_name=EMBED_MODEL)

def load_data_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        required_columns = ["Assessment Name", "URL", "Remote Testing Support",
                            "Adaptive/IRT Support", "Duration (min)", "Test Type"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

def build_index(data):
    Settings.embed_model = load_embeddings()
    Settings.llm = load_groq_llm()
    documents = [
        Document(text=f"Name: {item['Assessment Name']}, URL: {item['URL']}, Remote Testing: {item['Remote Testing Support']}, Adaptive/IRT: {item['Adaptive/IRT Support']}, Duration: {item['Duration (min)']}, Type: {item['Test Type']}")
        for item in data
    ]
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index

def load_chat_engine():
    if not os.path.exists(PERSIST_DIR):
        return None
    Settings.embed_model = load_embeddings()
    Settings.llm = load_groq_llm()
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    return index.as_chat_engine(chat_mode="context", verbose=True)

# --- Load or Build Index ---
try:
    chat_engine = load_chat_engine()
    if chat_engine is None:
        assessment_data = load_data_from_csv(CSV_FILE_PATH)
        build_index(assessment_data)
        chat_engine = load_chat_engine()
except Exception as e:
    print(f"❌ Error initializing chat engine: {e}")
    chat_engine = None

# --- Endpoint ---
@app.route("/assessments", methods=["POST"])
def get_assessments():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if chat_engine:
        try:
            response = chat_engine.chat(query)
            results = []

            for node in response.source_nodes:
                try:
                    parts = node.node.text.split(", ")
                    results.append({
                        "assessment_name": parts[0].split(": ")[1] if len(parts) > 0 else "N/A",
                        "assessment_url": parts[1].split(": ")[1] if len(parts) > 1 else "N/A",
                        "remote_testing_support": parts[2].split(": ")[1] if len(parts) > 2 else "N/A",
                        "adaptive_irt_support": parts[3].split(": ")[1] if len(parts) > 3 else "N/A",
                        "duration": parts[4].split(": ")[1] if len(parts) > 4 else "N/A",
                        "test_type": parts[5].split(": ")[1] if len(parts) > 5 else "N/A"
                    })
                except:
                    results.append({"error": "Error parsing assessment info"})

            return jsonify({"query": query, "response": results}), 200

        except Exception as e:
            return jsonify({"error": f"Chat processing error: {e}"}), 500
    else:
        return jsonify({"error": "Chat engine not initialized"}), 500

# --- Entry Point for Local Debugging ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
