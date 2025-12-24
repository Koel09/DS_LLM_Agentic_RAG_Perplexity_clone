import os
from typing import Dict, List, Literal

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch


# =========================
# CONFIG
# =========================

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_DIR = "data/chroma"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set")

# Groq via OpenAI-compatible API
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

web_search_tool = TavilySearch(api_key=TAVILY_API_KEY)


# =========================
# ROUTER
# =========================

Route = Literal["vectorstore", "websearch"]


def route_question(question: str) -> Route:
    prompt = f"""
Decide which source should answer the question.

Rules:
- If the question is about Sporo Health, its workflows, services,
  clinical processes, or internal documentation → vectorstore
- Otherwise → websearch

Output ONLY one word: vectorstore or websearch.

Question:
{question}
"""
    response = llm.invoke(prompt).content.strip().lower()
    return "vectorstore" if "vectorstore" in response else "websearch"


# =========================
# RETRIEVAL
# =========================

def retrieve_vectorstore(question: str, k: int = 5) -> str:
    docs = vectorstore.similarity_search(question, k=k)
    return "\n\n".join(d.page_content for d in docs)


def retrieve_web(question: str) -> str:
    return str(web_search_tool.invoke({"query": question}))


# =========================
# GRADERS
# =========================

def relevance_check(question: str, retrieved_text: str) -> bool:
    prompt = f"""
Determine if the retrieved text can plausibly answer the question.

Question:
{question}

Retrieved text:
{retrieved_text}

Output ONLY: yes or no
"""
    return llm.invoke(prompt).content.strip().lower() == "yes"


def hallucination_check(question: str, retrieved_text: str) -> bool:
    prompt = f"""
Is the retrieved text factually grounded and non-hallucinatory
with respect to the question?

Question:
{question}

Retrieved text:
{retrieved_text}

Output ONLY: yes or no
"""
    return llm.invoke(prompt).content.strip().lower() == "yes"


# =========================
# ANSWER GENERATION
# =========================

def answer_from_context(question: str, context: str) -> str:
    prompt = f"""
Answer the question using ONLY the provided context.

Question:
{question}

Context:
{context}

Rules:
- 2–3 sentences
- No external knowledge
- No hallucinations
"""
    return llm.invoke(prompt).content.strip()


def answer_from_web(question: str, web_text: str) -> str:
    prompt = f"""
Answer the question using ONLY the web search results.

Question:
{question}

Web results:
{web_text}

Rules:
- 2–3 sentences
- No external knowledge
"""
    return llm.invoke(prompt).content.strip()


# =========================
# MAIN PIPELINE
# =========================

def run_rag_pipeline(question: str) -> Dict:
    route = route_question(question)

    if route == "vectorstore":
        retrieved_text = retrieve_vectorstore(question)

        if relevance_check(question, retrieved_text) and hallucination_check(
            question, retrieved_text
        ):
            answer = answer_from_context(question, retrieved_text)
            return {
                "question": question,
                "route": "vectorstore",
                "answer": answer,
            }

        # fallback to web
        web_text = retrieve_web(question)
        answer = answer_from_web(question, web_text)
        return {
            "question": question,
            "route": "websearch_fallback",
            "answer": answer,
        }

    # direct web route
    web_text = retrieve_web(question)
    answer = answer_from_web(question, web_text)
    return {
        "question": question,
        "route": "websearch",
        "answer": answer,
    }

