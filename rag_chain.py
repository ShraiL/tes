from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

DB_DIR = 'vectorstore'
EMBEDDING_MODEL = 'nomic-embed-text'
LLM_MODEL = 'llama3.1'

emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model=LLM_MODEL)

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
    question = "What is machine learning?"
    print(f"Question: {question}")
    print(f"Answer: {ask(question)}")