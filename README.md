# ML Final Exam Templates 🎯

---

## 📦 Quick Setup

### Step 1: Install Packages

**Option A: RAG Only (Faster)**
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb streamlit
```

**Option B: RAG + BLIP (Full)**
```bash
pip install -r requirements-full.txt
```
Or manually:
```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb streamlit transformers pillow torch
```

### Step 2: Download Ollama Models
```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

### Step 3: Start Ollama Server (Keep Running!)
```bash
ollama serve
```

---

## 📁 Folder Structure
```
ML_Final_Exam/
├── data/                  ← PUT .TXT FILES HERE
│   └── sample.txt
├── images/                ← PUT IMAGES HERE (for BLIP)
├── ingest.py              ← Step 1: Load documents
├── rag_chain.py           ← (Optional) Test in terminal
├── app.py                 ← Step 2: Streamlit web app
├── blip_caption.py        ← BLIP image-to-text
├── requirements.txt       ← RAG only
├── requirements-full.txt  ← RAG + BLIP
└── README.md
```

---

## 📝 TASK 1: RAG Pipeline (40 Points)

### Step 1: Put exam's .txt files in `data/` folder

### Step 2: Run ingest
```bash
python ingest.py
```
Output: "Loaded X documents" → "Created Y chunks" → "Done!"

### Step 3: Run Streamlit
```bash
streamlit run app.py
```
Browser opens → Type question → Click button → Get answer!

---

## 🖼️ TASK 2: BLIP (Backup)

### Step 1: Put image in `images/` folder

### Step 2: Change `IMAGE_PATH` in `blip_caption.py`
```python
IMAGE_PATH = r"images/your_image.jpg"
```

### Step 3: Run
```bash
python blip_caption.py
```

---

## 🔧 What To Change

### ingest.py
| Variable | Default | When To Change |
|----------|---------|----------------|
| `DATA_DIR` | `'data'` | Professor says different folder |
| `EMBEDDING_MODEL` | `'nomic-embed-text'` | Professor specifies model |
| `CHUNK_SIZE` | `2000` | Professor specifies size |
| `CHUNK_OVERLAP` | `50` | Professor specifies overlap |

### app.py & rag_chain.py
| Variable | Default | When To Change |
|----------|---------|----------------|
| `EMBEDDING_MODEL` | `'nomic-embed-text'` | ⚠️ Must match ingest.py! |
| `LLM_MODEL` | `'llama3.1'` | Professor specifies model |
| `NUM_RESULTS` | `3` | Professor specifies |

### blip_caption.py
| Variable | Default | When To Change |
|----------|---------|----------------|
| `IMAGE_PATH` | `'images/test.jpg'` | Always - your image path! |
| `TEXT_PROMPT` | `'a photo of'` | Professor specifies |
| `QUESTION` | `'What is in the image?'` | Your question |

---

## 🔄 Possible Professor Changes

| Professor Says | What To Change |
|----------------|----------------|
| "Use folder 'documents'" | `DATA_DIR = 'documents'` in ingest.py |
| "Use llama3.1 for embeddings" | `EMBEDDING_MODEL = 'llama3.1'` in ALL 3 files! |
| "Chunk size 1000" | `CHUNK_SIZE = 1000` in ingest.py |
| "Chunk overlap 200" | `CHUNK_OVERLAP = 200` in ingest.py |
| "Retrieve 5 documents" | `NUM_RESULTS = 5` in app.py & rag_chain.py |
| "Use llama3.2 for answers" | `LLM_MODEL = 'llama3.2'` in app.py & rag_chain.py |

---

## ⚠️ Important Rules

| Rule | Why |
|------|-----|
| `EMBEDDING_MODEL` must be SAME in all 3 files | Different = ERROR! |
| Run `ollama serve` before running code | Connection error without it |
| Don't name files `ollama.py` or `transformers.py` | Circular import error |
| Use `r""` for Windows paths | `r"C:\path\file.txt"` |

---

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| "Connection refused" | Run `ollama serve` in terminal |
| "Model not found" | Run `ollama pull model_name` |
| "No module named..." | Run `pip install -r requirements.txt` |
| Circular import error | Rename your file (not ollama.py!) |
| "I don't know" answer | Ask question about content IN your document |

---

## ✅ Exam Day Checklist

- [ ] Clone repo
- [ ] `pip install -r requirements.txt` (or `requirements-full.txt` for BLIP)
- [ ] `ollama serve` (keep terminal open!)
- [ ] Put exam's .txt file in `data/` folder
- [ ] `python ingest.py`
- [ ] `streamlit run app.py`
- [ ] Test with a question
- [ ] Show professor! 🎉

---

## 📋 Quick Commands Summary

```bash
# Install packages (choose one)
pip install -r requirements.txt           # RAG only
pip install -r requirements-full.txt      # RAG + BLIP

# Download models
ollama pull nomic-embed-text
ollama pull llama3.1

# Start Ollama (keep running!)
ollama serve

# RAG Pipeline
python ingest.py
streamlit run app.py

# BLIP (if needed)
python blip_caption.py
```

---

Good luck! 🍀
