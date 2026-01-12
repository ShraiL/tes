import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

DB_DIR = 'vectorstore'
EMBEDDING_MODEL = 'nomic-embed-text'
LLM_MODEL = 'llama3.1'

@st.cache_resource
def load_vectorstore():
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    return db

@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL)

template = PromptTemplate(
    template="""Use the following context to answer the question.
If you don't know the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def ask(question, db, llm):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = template.format(context=context, question=question)
    answer = llm.invoke(formatted_prompt)
    return answer

st.title("RAG Document Q&A")
st.write("Ask questions about the loaded documents")

db = load_vectorstore()
llm = load_llm()

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching..."):
            answer = ask(question, db, llm)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question!")