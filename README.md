# ML Final Exam Templates 🎯

## Quick Setup
```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb streamlit transformers pillow torch
ollama pull nomic-embed-text
ollama pull llama3.1
```

---

## TASK 1: RAG Pipeline

### Step 1: Put .txt files in `data/` folder

### Step 2: Run ingest
```bash
python ingest.py
```

### Step 3: Run streamlit
```bash
streamlit run app.py
```

---

## TASK 2: BLIP

### Step 1: Change `IMAGE_PATH` in `blip_caption.py`

### Step 2: Run
```bash
python blip_caption.py
```

---

## 🔧 What To Change

### ingest.py
| Variable | When To Change |
|----------|----------------|
| `DATA_DIR` | Professor says different folder |
| `EMBEDDING_MODEL` | Professor specifies model |
| `CHUNK_SIZE` | Professor specifies size |

### app.py & rag_chain.py
| Variable | When To Change |
|----------|----------------|
| `EMBEDDING_MODEL` | Must match ingest.py! |
| `LLM_MODEL` | Professor specifies model |

### blip_caption.py
| Variable | When To Change |
|----------|----------------|
| `IMAGE_PATH` | Always - your image! |
| `TEXT_PROMPT` | Professor specifies |
| `QUESTION` | Your question |

---

## ⚠️ Remember
- `EMBEDDING_MODEL` must be SAME in all 3 files!
- Run `ollama serve` before running code!
- Don't name files `ollama.py` or `transformers.py`
