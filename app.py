import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  🔧 CHANGE THESE IF PROFESSOR ASKS                                          ║
# ╠═════════════════════════════════════════════════════════════════════════════╣
# ║  ⚠️  IMPORTANT: These MUST match ingest.py!                                 ║
# ║  DB_DIR           → Must match ingest.py!                                   ║
# ║  EMBEDDING_MODEL  → Must match ingest.py!                                   ║
# ║  LLM_MODEL        → Model for generating answers                            ║
# ║  NUM_RESULTS      → How many chunks to retrieve                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

DB_DIR = 'vectorstore'                # ← Must match ingest.py!
EMBEDDING_MODEL = 'nomic-embed-text'  # ← Must match ingest.py!
LLM_MODEL = 'llama3.1'                # ← CHANGE if professor says different model
NUM_RESULTS = 3                       # ← CHANGE if professor specifies


@st.cache_resource
def load_vectorstore():
    emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
    return db


@st.cache_resource
def load_llm():
    return Ollama(model=LLM_MODEL)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  📝 PROMPT TEMPLATE - Change the text inside template="" if professor asks  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
template = PromptTemplate(
    template="""Use the following context to answer the question.
If you don't know the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


def ask(question, db, llm):
    retriever = db.as_retriever(search_kwargs={"k": NUM_RESULTS})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = template.format(context=context, question=question)
    answer = llm.invoke(formatted_prompt)
    return answer


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  🖥️  STREAMLIT UI - Change title/text if professor asks                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

st.title("RAG Document Q&A")                              # ← CHANGE title if needed
st.write("Ask questions about the loaded documents")      # ← CHANGE description if needed

db = load_vectorstore()
llm = load_llm()

question = st.text_input("Enter your question:")          # ← CHANGE label if needed

if st.button("Get Answer"):                               # ← CHANGE button text if needed
    if question:
        with st.spinner("Searching..."):
            answer = ask(question, db, llm)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question!")