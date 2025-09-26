# 📚 RAG-Powered PDF Chatbot  

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload **PDF documents** and interact with them conversationally. Ask questions, summarize sections, or extract insights — all grounded in your actual documents.  

---

## 🌟 What Makes This Special  

Unlike generic chatbots that rely only on pre-trained data, this RAG-powered system:  

- 📄 **Uses YOUR PDFs** – all responses are grounded in uploaded files  
- 🔍 **Efficient retrieval** – only relevant chunks of text are used, removing token limit issues  
- ✨ **Authentic answers** – chatbot never “hallucinates” outside your documents  
- 🧠 **Semantic search** – understands your question even when phrased differently  

---

## 🎯 How It Works  

1. **Upload PDF(s)**  
   → Extract text  
   → Split into chunks  
   → Generate embeddings  

2. **Vector Store Search**  
   → Store chunks in FAISS/Chroma  
   → Retrieve the most relevant sections  

3. **Answer with AI**  
   → Combine retrieved chunks  
   → Use LLM (OpenAI, Ollama, HuggingFace) to generate a grounded response  

---

## ⚡ Features  

- 💬 Chat with your PDFs in real time  
- 📑 Summarization of uploaded files  
- 🔗 Multi-document support (ask across PDFs)  
- 📂 Local-first (no need for external DB like Supabase)  
- 🎨 Streamlit-based simple, interactive UI  

---

## 🛠️ Tech Stack  

- **Frontend/UI** → Streamlit  
- **RAG Infrastructure** → LangChain  
- **Vector Store** → FAISS / ChromaDB (local)  
- **Embeddings & LLMs** → OpenAI / HuggingFace / Ollama  
- **PDF Processing** → PyPDF2 / pdfplumber  

---

## 🚀 Quick Start  

### 1. Prerequisites  
- Python 3.10+  
- OpenAI API key (if using GPT models)  

### 2. Setup  
```bash
git clone https://github.com/your-username/pdf-chatbot.git
cd pdf-chatbot
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
### 3. Run the App
```bash
streamlit run app_local.py
```
### 📊 System Architecture
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Store
      ↓                                             ↑
      └───────────── User Query → Vector Search → LLM Answer

### 🎯 Use Cases

- Students & Researchers → Summarize academic papers, extract definitions

- Business Analysts → Query reports, contracts, or financial documents

- Professionals → Quickly pull insights from compliance docs, whitepapers

### 🤝 Contributing

This is an evolving project — contributions are welcome! Feel free to fork and submit PRs.


Do you also want me to generate a **ready-to-use `requirements.txt`** from your `app_local.py` so that whoever clones your repo can run it without setup issues?
