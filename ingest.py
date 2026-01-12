import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  🔧 CHANGE THESE IF PROFESSOR ASKS                                          ║
# ╠═════════════════════════════════════════════════════════════════════════════╣
# ║  DATA_DIR         → Folder where .txt files are located                     ║
# ║  DB_DIR           → Folder where ChromaDB will save (usually keep as is)    ║
# ║  EMBEDDING_MODEL  → 'nomic-embed-text' or 'llama3.1' (professor will say)   ║
# ║  CHUNK_SIZE       → How big each chunk is (default 2000)                    ║
# ║  CHUNK_OVERLAP    → Overlap between chunks (default 50)                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

DATA_DIR = 'data'                     # ← CHANGE if professor says different folder
DB_DIR = 'vectorstore'                # ← Usually keep this
EMBEDDING_MODEL = 'nomic-embed-text'  # ← CHANGE if professor says different model
CHUNK_SIZE = 2000                     # ← CHANGE if professor specifies
CHUNK_OVERLAP = 50                    # ← CHANGE if professor specifies


def load_doc():
    docs = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.txt'):
            path = os.path.join(DATA_DIR, f)
            loader = TextLoader(path, autodetect_encoding=True)
            docs.extend(loader.load())
    print(f'Loaded {len(docs)} documents')
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f'Created {len(chunks)} chunks')
    return chunks


def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    db.persist()
    print(f'Vectorstore saved!')


if __name__ == '__main__':
    docs = load_doc()
    chunks = split_docs(docs)
    create_vectorstore(chunks)
    print('Done!')