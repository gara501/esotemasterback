import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder


load_dotenv()

COLLECTION_NAME = "esoteric_books"

LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
RERANKER_MODEL = "BAAI/bge-reranker-base"


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing Google AI Studio API key. Set GOOGLE_API_KEY or GEMINI_API_KEY."
        )

    return api_key


MODE_PROMPTS = {
    "general": """
Actúa como un maestro esotérico multidisciplinario.
Puedes utilizar hermetismo, simbolismo, alquimia, tarot, cábala,
gnosticismo, mitología comparada y psicología simbólica.
""",

    "extract": """
Actúa como un buscador textual dentro de la biblioteca.
Tu tarea NO es interpretar ni analizar.
Tu tarea es localizar y devolver el fragmento textual más relevante.
""",

    "tarot": """
Actúa como un maestro especializado en Tarot.
Enfócate en arcanos, simbolismo tradicional, Tarot de Marsella,
Rider-Waite y lectura arquetípica, no adivinatoria.
""",

    "cabala": """
Actúa como un maestro especializado en Cábala y Cábala hermética.
Enfócate en Árbol de la Vida, Sephiroth, senderos y correspondencias herméticas.
""",

    "alquimia": """
Actúa como un maestro especializado en alquimia.
Enfócate en Nigredo, Albedo, Citrinitas, Rubedo, Mercurio, Azufre,
Sal, Piedra filosofal y transmutación interior.
""",

    "comparative": """
Actúa como un investigador comparativo de sistemas esotéricos.
Debes identificar similitudes, diferencias, influencias históricas,
correspondencias simbólicas y tensiones entre tradiciones.
""",
}


DEFAULT_RAG_CONFIG = {
    "use_rerank": True,
    "initial_k": 20,
    "final_k": 8,
    "num_predict": 4096,
    "detail_level": "extensive",
    "force_detailed_answer": True,
}


google_api_key = get_google_api_key()

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=google_api_key,
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    chroma_cloud_api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
    create_collection_if_not_exists=False,
)

reranker = CrossEncoder(RERANKER_MODEL)


prompt = ChatPromptTemplate.from_template("""
Eres un maestro de estudios esotéricos.

Responde en español usando SOLO el contexto de los libros proporcionados.

Modo activo:
{mode_prompt}

Nivel de detalle solicitado:
{detail_level}

Reglas obligatorias:
- Devuelve SOLO JSON válido.
- No uses markdown.
- No envuelvas la respuesta en ```json.
- No inventes información fuera del contexto.
- No inventes fuentes.
- No inventes páginas.
- Si el contexto no alcanza, dilo con claridad.
- No respondas en un solo párrafo.
- La respuesta debe ser larga, profunda, pedagógica y estructurada.
- Usa todo el contexto relevante disponible.
- Desarrolla cada idea con detalle.
- Cada sección importante debe tener varios párrafos.
- Explica conceptos abstractos con ejemplos cuando el contexto lo permita.
- Relaciona símbolos, historia, tradición y práctica esotérica cuando aparezcan en las fuentes.
- No presentes afirmaciones místicas como hechos científicos.
- No resumas demasiado.

Pregunta del usuario:
{question}

Contexto de libros:
{context}

Formato JSON obligatorio:
{{
  "title": "...",
  "mode": "{mode}",
  "short_answer": "...",
  "historical_context": "...",
  "symbolic_interpretation": "...",
  "expanded_analysis": "...",
  "deep_explanation": "...",
  "key_concepts": ["..."],
  "reflection": "...",
  "sources_used": [
    {{
      "book": "...",
      "page": "..."
    }}
  ]
}}
""")


def build_llm(num_predict: int = 4096) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=google_api_key,
        temperature=0.25,
        max_output_tokens=num_predict,
    )


def normalize_rag_config(rag_config: dict | None) -> dict:
    config = DEFAULT_RAG_CONFIG.copy()

    if rag_config:
        config.update(rag_config)

    config["initial_k"] = int(config.get("initial_k", 20))
    config["final_k"] = int(config.get("final_k", 8))
    config["num_predict"] = int(config.get("num_predict", 4096))
    config["use_rerank"] = bool(config.get("use_rerank", True))
    config["force_detailed_answer"] = bool(config.get("force_detailed_answer", True))

    if config["final_k"] > config["initial_k"]:
        config["final_k"] = config["initial_k"]

    return config


def build_context(docs: list[Any]) -> str:
    return "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Fuente desconocida')} "
            f"page {doc.metadata.get('page', 'N/A')}\n"
            f"{doc.page_content}"
        )
        for doc in docs
    )


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def doc_to_extract(doc: Any) -> dict:
    return {
        "book": doc.metadata.get("source", "Fuente desconocida"),
        "page": doc.metadata.get("page", "N/A"),
        "text": doc.page_content.strip(),
    }


def rerank_documents(query: str, docs: list[Any], top_n: int = 8) -> list[Any]:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda item: item[1], reverse=True)

    return [doc for doc, _score in scored_docs[:top_n]]


def retrieve_documents(
    question: str,
    mode: str,
    rag_config: dict,
) -> list[Any]:
    search_query = f"{mode}: {question}"

    initial_docs = vectorstore.similarity_search(
        search_query,
        k=rag_config["initial_k"],
    )

    if not rag_config["use_rerank"]:
        return initial_docs[:rag_config["final_k"]]

    return rerank_documents(
        query=question,
        docs=initial_docs,
        top_n=rag_config["final_k"],
    )


def retrieve_extract_documents(question: str, k: int = 20) -> list[Any]:
    return vectorstore.similarity_search(question, k=k)


def extract_text(question: str, rag_config: dict | None = None) -> str:
    config = normalize_rag_config(rag_config)

    docs = retrieve_extract_documents(
        question=question,
        k=config["initial_k"],
    )

    if not docs:
        result = {
            "title": "No se encontraron extractos",
            "mode": "extract",
            "found": False,
            "matched_query": question,
            "extracts": [],
            "notes": "No se encontraron fragmentos relevantes en Chroma Cloud.",
        }

        return json.dumps(result, ensure_ascii=False)

    if config["use_rerank"]:
        docs = rerank_documents(
            query=question,
            docs=docs,
            top_n=config["final_k"],
        )
    else:
        docs = docs[:config["final_k"]]

    normalized_question = normalize_text(question)

    exact_matches = []
    semantic_matches = []

    for doc in docs:
        normalized_content = normalize_text(doc.page_content)

        if normalized_question and normalized_question in normalized_content:
            exact_matches.append(doc)
        else:
            semantic_matches.append(doc)

    selected_docs = exact_matches if exact_matches else semantic_matches

    extracts = [doc_to_extract(doc) for doc in selected_docs[:config["final_k"]]]

    result = {
        "title": "Extracto textual encontrado",
        "mode": "extract",
        "found": len(extracts) > 0,
        "matched_query": question,
        "extracts": extracts,
        "notes": (
            "Se devuelven fragmentos textuales recuperados de Chroma Cloud. "
            "Si el texto pertenece a una sección larga, puede aparecer dividido en varios fragmentos."
        ),
    }

    return json.dumps(result, ensure_ascii=False)


def answer_question(
    question: str,
    mode: str = "general",
    rag_config: dict | None = None,
) -> str:
    selected_mode = mode if mode in MODE_PROMPTS else "general"
    config = normalize_rag_config(rag_config)

    if selected_mode == "extract":
        return extract_text(question, config)

    docs = retrieve_documents(
        question=question,
        mode=selected_mode,
        rag_config=config,
    )

    if not docs:
        result = {
            "title": "No se encontró información suficiente",
            "mode": selected_mode,
            "short_answer": "No encontré fragmentos relevantes en Chroma Cloud para responder esta pregunta.",
            "historical_context": "",
            "symbolic_interpretation": "",
            "expanded_analysis": "",
            "deep_explanation": "",
            "key_concepts": [],
            "reflection": "",
            "sources_used": [],
        }

        return json.dumps(result, ensure_ascii=False)

    context = build_context(docs)

    llm = build_llm(num_predict=config["num_predict"])
    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "context": context,
        "mode": selected_mode,
        "mode_prompt": MODE_PROMPTS[selected_mode],
        "detail_level": config["detail_level"],
    })

    return response.content
