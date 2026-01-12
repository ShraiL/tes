# ML Final Exam

## Setup
```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb streamlit transformers pillow torch
ollama pull nomic-embed-text
ollama pull llama3.1
```

## RAG Pipeline
1. Put .txt files in `data/` folder
2. Run: `python ingest.py`
3. Run: `streamlit run app.py`

## BLIP
1. Change `IMAGE_PATH` in `blip_caption.py`
2. Run: `python blip_caption.py`
