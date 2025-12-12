import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
import uuid
from pymongo import MongoClient
import numpy as np
import config 

# MongoDB Atlas Setup
MONGO_URI = config.MONGO_URI

if not MONGO_URI:
    st.error("MONGO_URI environment variable is missing!")
    st.stop()

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["pdf_vectors_db"]
collection = db["vectors"]

# Embedding Model
@st.cache_resource
def load_embedder():
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# HuggingFace LLM
@st.cache_resource
def load_llm():
    MODEL_NAME = config.LLM_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def add_vectors(embeddings, metadatas):
    docs = []
    for emb, meta in zip(embeddings, metadatas):
        emb_norm = normalize(emb)
        docs.append({
            "embedding": emb_norm.tolist(),
            "metadata": meta
        })
    if docs:
        collection.insert_many(docs)

def mongodb_vector_search(query_vector, k=3):
    query_vector = normalize(query_vector)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector.tolist(),
                "numCandidates": 100,
                "limit": k
            }
        },
        {"$project": {"metadata": 1, "_id": 0}}
    ]
    results = list(collection.aggregate(pipeline))
    return [r["metadata"] for r in results]

def generate_answer(model, tokenizer, context, question):
    prompt = f"""
You are a helpful assistant. Use ONLY the context below.

Context:
{context}

Question: {question}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2
    )
    out = tokenizer.decode(output[0], skip_special_tokens=True)
    return out.split("Answer:")[-1].strip()

def main():
    st.title("PDF Files RAG Chatbot")
    st.subheader("HuggingFace + MongoDB Atlas Vector Search")

    st.markdown(f"Loaded LLM model: **{config.LLM_MODEL_NAME}**")

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        st.markdown(f"Running on **CUDA GPU: {device_name}**")
    else:
        st.markdown("Running on **CPU**")

    existing_files = collection.distinct("metadata.file_name")
    if existing_files:
        st.markdown("#### 📁 Already Indexed PDF Files:")
        container = st.container()
        container.markdown(
            """
            <div style='max-height: 150px; overflow-y: auto; border: 1px solid #ddd; padding: 8px;'>
            """ + "\n".join(f"- {fname}" for fname in existing_files) + """
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("No PDFs indexed yet. Upload files to start.")

    st.markdown("###  Delete")
    if st.button("Delete ALL Indexed PDF Files", type="primary"):
        collection.delete_many({})
        st.session_state.indexed_files = set()
        st.session_state.shown_warnings = set()
        st.success("🔥 All files and embeddings deleted successfully!")
        st.rerun()

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()
    if "shown_warnings" not in st.session_state:
        st.session_state.shown_warnings = set()

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        embedder = load_embedder()

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            existing_count = collection.count_documents({"metadata.file_name": file_name})
            if existing_count > 0:
                if file_name not in st.session_state.shown_warnings:
                    st.warning(
                        f"⚠️ Skipping **{file_name}** — already uploaded and indexed ({existing_count} chunks found)."
                    )
                    st.session_state.shown_warnings.add(file_name)
                continue

            pdf_id = str(uuid.uuid4())
            pdf = PdfReader(uploaded_file)
            pages = [page.extract_text() for page in pdf.pages]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )

            chunks = splitter.create_documents(pages)
            st.write(f"📄 **{file_name}** → {len(chunks)} chunks")

            embeddings = embedder.encode([c.page_content for c in chunks])

            metadatas = []
            for idx, chunk in enumerate(chunks):
                metadatas.append({
                    "pdf_id": pdf_id,
                    "file_name": file_name,
                    "page_number": chunk.metadata.get("page", None),
                    "chunk_id": idx,
                    "text": chunk.page_content
                })

            add_vectors(embeddings, metadatas)
            st.session_state.indexed_files.add(file_name)
            st.success(f"✅ {file_name} embedded and stored in MongoDB.")

    question = st.text_input("Ask a question:")

    if question:
        tokenizer, model = load_llm()
        embedder = load_embedder()

        query_vector = embedder.encode([question])[0]
        results = mongodb_vector_search(query_vector, k=3)

        if not results:
            st.error("No relevant content found in your indexed PDFs.")
            return

        context = "\n\n".join(r["text"] for r in results)
        answer = generate_answer(model, tokenizer, context, question)

        st.markdown("### 🧠 Answer")
        st.write(answer)

        st.markdown("### 📌 Sources")
        for src in results:
            st.write(
                f"- **{src['file_name']}** | Page: {src['page_number']} | Chunk: {src['chunk_id']}"
            )

if __name__ == "__main__":
    main()

