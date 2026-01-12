ML Final Exam
Setup
bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb streamlit transformers pillow torch
ollama pull nomic-embed-text
ollama pull llama3.1
RAG Pipeline
Put .txt files in data/ folder
Run: python ingest.py
Run: streamlit run app.py
BLIP
Change IMAGE_PATH in blip_caption.py
Run: python blip_caption.py
Your GitHub Repository Structure:
ML_Final_Exam/
├── data/
│   └── sample.txt       ← Copy sample.txt here
├── images/              ← Create empty folder (for BLIP images)
├── ingest.py
├── rag_chain.py
├── app.py
├── blip_caption.py
└── README.md

On Exam Day:
StepCommand1. Put exam's .txt file in data/2. Run ingestpython ingest.py3. Run streamlitstreamlit run app.py
