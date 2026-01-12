from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  🔧 CHANGE THESE IF PROFESSOR ASKS                                          ║
# ╠═════════════════════════════════════════════════════════════════════════════╣
# ║  DB_DIR           → Must match ingest.py!                                   ║
# ║  EMBEDDING_MODEL  → Must match ingest.py!                                   ║
# ║  LLM_MODEL        → Model for generating answers ('llama3.1' or 'llama3.2') ║
# ║  NUM_RESULTS      → How many chunks to retrieve (default 3)                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

DB_DIR = 'vectorstore'                # ← Must match ingest.py!
EMBEDDING_MODEL = 'nomic-embed-text'  # ← Must match ingest.py!
LLM_MODEL = 'llama3.1'                # ← CHANGE if professor says different model
NUM_RESULTS = 3                       # ← CHANGE if professor specifies


# Load vectorstore
emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=emb)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": NUM_RESULTS})

# Load LLM
llm = Ollama(model=LLM_MODEL)

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


def ask(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = template.format(context=context, question=question)
    answer = llm.invoke(formatted_prompt)
    return answer


if __name__ == '__main__':
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  ❓ CHANGE YOUR QUESTION HERE FOR TESTING                             ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    question = "What is machine learning?"  # ← CHANGE to test different questions
    
    print(f"Question: {question}")
    print(f"Answer: {ask(question)}")