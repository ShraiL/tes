import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = 'data'
DB_DIR = 'vectorstore'
EMBEDDING_MODEL = 'nomic-embed-text'
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 50

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