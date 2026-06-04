import json
import logging
import os
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

from src.cache import get_cached_response, set_cached_response, stable_cache_key
from src.qdrant_store import (
    build_metadata_filter,
    detect_embedding_dimension,
    ensure_collection,
    get_embeddings,
    get_qdrant_client,
    get_settings,
)


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Eres un asistente experto en análisis de libros y documentos. Responde "
    "únicamente usando el contexto proporcionado. No inventes información. Si "
    "el contexto no contiene evidencia suficiente, dilo claramente. Explica "
    "con detalle, conecta ideas entre fragmentos y cita las fuentes por libro "
    "y página. Prioriza precisión sobre creatividad."
)

MODE_PROMPTS = {
    "general": """
Actúa como un maestro esotérico multidisciplinario.
Puedes utilizar hermetismo, simbolismo, alquimia, tarot, cábala,
gnosticismo, satanismo, mitología comparada y psicología simbólica.
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
    "use_rerank": None,
    "initial_k": None,
    "fetch_k": 40,
    "final_k": None,
    "num_predict": 8192,
    "detail_level": "extensive",
    "force_detailed_answer": True,
    "debug_chunks": False,
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
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
- La respuesta debe ser profunda, pedagógica y estructurada.
- Cita fuentes por libro y página cuando uses una idea.

Pregunta del usuario:
{question}

Contexto de libros:
{context}

Formato JSON obligatorio:
{{
  "title": "...",
  "mode": "{mode}",
  "answer": "...",
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
""",
        ),
    ]
)

_reranker = None


@lru_cache(maxsize=1)
def get_vectorstore() -> QdrantVectorStore:
    settings = get_settings()
    embeddings = get_embeddings(settings)
    vector_size = detect_embedding_dimension(embeddings)
    client = get_qdrant_client(settings)
    ensure_collection(client, settings.collection, vector_size)

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection,
        embedding=embeddings,
        content_payload_key="content",
        metadata_payload_key="metadata",
    )


def build_llm(num_predict: int = 8192) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.google_llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.25,
        max_output_tokens=num_predict,
        response_mime_type="application/json",
    )


def normalize_rag_config(rag_config: dict | None) -> dict:
    settings = get_settings()
    config = DEFAULT_RAG_CONFIG.copy()
    if rag_config:
        config.update(rag_config)

    if config.get("initial_k") is None:
        config["initial_k"] = settings.retrieval_k
    if config.get("final_k") is None:
        config["final_k"] = settings.final_top_n
    if config.get("use_rerank") is None:
        config["use_rerank"] = settings.use_reranker

    config["initial_k"] = int(config.get("initial_k", 20))
    config["fetch_k"] = int(config.get("fetch_k", 40))
    config["final_k"] = int(config.get("final_k", 8))
    config["num_predict"] = int(config.get("num_predict", 8192))
    config["use_rerank"] = bool(config.get("use_rerank")) and settings.use_reranker
    config["force_detailed_answer"] = bool(config.get("force_detailed_answer", True))
    config["debug_chunks"] = bool(config.get("debug_chunks", False))

    if config["final_k"] > config["initial_k"]:
        config["final_k"] = config["initial_k"]

    if config["fetch_k"] < config["initial_k"]:
        config["fetch_k"] = config["initial_k"]

    return config


def build_response_cache_key(
    question: str,
    mode: str,
    rag_config: dict,
    filters: dict[str, Any] | None,
) -> str:
    settings = get_settings()
    return stable_cache_key(
        "rag_response",
        {
            "question": normalize_text(question),
            "mode": mode,
            "rag_config": rag_config,
            "filters": filters or {},
            "collection": settings.collection,
            "llm_model": settings.google_llm_model,
            "embedding_model": settings.google_embedding_model,
            "reranker_model": settings.reranker_model,
            "cache_version": os.getenv("RAG_CACHE_VERSION", "1"),
        },
    )


def build_context(docs: list[Document]) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        parts.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"Libro: {metadata.get('title', 'Fuente desconocida')}",
                    f"Archivo: {metadata.get('file_name', metadata.get('source', 'N/A'))}",
                    f"Página: {metadata.get('page', 'N/A')}",
                    f"Chunk ID: {metadata.get('chunk_id', 'N/A')}",
                    "Texto:",
                    doc.page_content,
                ]
            )
        )
    return "\n\n".join(parts)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def parse_json_payload(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a complete JSON object.")

    return json.loads(cleaned[start : end + 1])


def fallback_payload(mode: str, message: str) -> dict:
    return {
        "title": "No se encontró información suficiente",
        "mode": mode,
        "answer": message,
        "short_answer": message,
        "historical_context": "",
        "symbolic_interpretation": "",
        "expanded_analysis": "",
        "deep_explanation": "",
        "key_concepts": [],
        "reflection": "",
        "sources_used": [],
        "grouped_sources": [],
    }


def model_payload(text: str, mode: str) -> dict:
    try:
        parsed = parse_json_payload(text)
        if not isinstance(parsed, dict):
            raise ValueError("Model JSON response is not an object.")
        return parsed
    except Exception:
        return fallback_payload(
            mode,
            (
                "El modelo devolvió una respuesta incompleta o no válida como "
                "JSON. Intenta de nuevo con una pregunta más específica o "
                "reduce el alcance de la consulta."
            ),
        )


def doc_to_source(doc: Document) -> dict:
    return {
        "book": doc.metadata.get("title")
        or doc.metadata.get("source")
        or "Fuente desconocida",
        "page": doc.metadata.get("page", "N/A"),
    }


def doc_to_extract(doc: Document) -> dict:
    return {
        "book": doc.metadata.get("title")
        or doc.metadata.get("source")
        or "Fuente desconocida",
        "page": doc.metadata.get("page", "N/A"),
        "file_name": doc.metadata.get("file_name"),
        "chunk_id": doc.metadata.get("chunk_id"),
        "text": doc.page_content.strip(),
    }


def group_sources(docs: list[Document]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[int]] = {}

    for doc in docs:
        title = doc.metadata.get("title") or "Fuente desconocida"
        file_name = doc.metadata.get("file_name") or doc.metadata.get("source") or ""
        key = (title, file_name)
        grouped.setdefault(key, set())

        page = doc.metadata.get("page")
        try:
            grouped[key].add(int(page))
        except (TypeError, ValueError):
            continue

    return [
        {
            "title": title,
            "file_name": file_name,
            "pages": sorted(pages),
        }
        for (title, file_name), pages in grouped.items()
    ]


def debug_chunks(docs: list[Document]) -> list[dict[str, Any]]:
    return [
        {
            "title": doc.metadata.get("title"),
            "file_name": doc.metadata.get("file_name"),
            "page": doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "content_hash": doc.metadata.get("content_hash"),
            "text_preview": doc.page_content[:400],
        }
        for doc in docs
    ]


def get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker

    settings = get_settings()
    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.reranker_model)
        return _reranker
    except Exception as exc:
        logger.warning("Reranker unavailable; using Qdrant order. Error: %s", exc)
        return None


def rerank_documents(query: str, docs: list[Document], top_n: int) -> list[Document]:
    if not docs:
        return []

    reranker = get_reranker()
    if reranker is None:
        return docs[:top_n]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda item: float(item[1]), reverse=True)
    return [doc for doc, _score in scored_docs[:top_n]]


def merge_documents(*doc_groups: list[Document]) -> list[Document]:
    merged = []
    seen = set()
    for docs in doc_groups:
        for doc in docs:
            key = doc.metadata.get("chunk_id") or (
                doc.metadata.get("file_name"),
                doc.metadata.get("page"),
                doc.page_content[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return merged


def retrieve_documents(
    question: str,
    mode: str,
    rag_config: dict,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    vectorstore = get_vectorstore()
    qdrant_filter = build_metadata_filter(filters)
    search_query = f"{mode}: {question}"

    similarity_docs = vectorstore.similarity_search(
        search_query,
        k=rag_config["initial_k"],
        filter=qdrant_filter,
    )

    try:
        mmr_docs = vectorstore.max_marginal_relevance_search(
            search_query,
            k=rag_config["initial_k"],
            fetch_k=rag_config["fetch_k"],
            lambda_mult=0.5,
            filter=qdrant_filter,
        )
    except Exception as exc:
        logger.warning("MMR search unavailable; using similarity only. Error: %s", exc)
        mmr_docs = []

    candidates = merge_documents(similarity_docs, mmr_docs)

    if rag_config["use_rerank"]:
        return rerank_documents(
            query=question,
            docs=candidates,
            top_n=rag_config["final_k"],
        )

    return candidates[: rag_config["final_k"]]


def retrieve_extract_documents(
    question: str,
    k: int = 20,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(
        question,
        k=k,
        filter=build_metadata_filter(filters),
    )


def extract_text(
    question: str,
    rag_config: dict | None = None,
    filters: dict[str, Any] | None = None,
) -> str:
    config = normalize_rag_config(rag_config)
    cache_key = build_response_cache_key(question, "extract", config, filters)
    cached_response = get_cached_response(cache_key)
    if cached_response is not None:
        logger.info("RAG response cache hit for extract query.")
        return cached_response

    docs = retrieve_extract_documents(
        question=question,
        k=config["initial_k"],
        filters=filters,
    )

    if config["use_rerank"]:
        docs = rerank_documents(question, docs, config["final_k"])
    else:
        docs = docs[: config["final_k"]]

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
    extracts = [doc_to_extract(doc) for doc in selected_docs[: config["final_k"]]]

    result = {
        "title": "Extracto textual encontrado" if extracts else "No se encontraron extractos",
        "mode": "extract",
        "found": len(extracts) > 0,
        "matched_query": question,
        "answer": "",
        "extracts": extracts,
        "sources_used": [doc_to_source(doc) for doc in selected_docs],
        "grouped_sources": group_sources(selected_docs),
        "notes": (
            "Se devuelven fragmentos textuales recuperados de Qdrant Cloud. "
            "Si el texto pertenece a una sección larga, puede aparecer dividido "
            "en varios fragmentos."
        ),
    }

    if config["debug_chunks"]:
        result["debug_chunks"] = debug_chunks(selected_docs)

    response_json = json.dumps(result, ensure_ascii=False)
    set_cached_response(cache_key, response_json)
    return response_json


def answer_question(
    question: str,
    mode: str = "general",
    rag_config: dict | None = None,
    filters: dict[str, Any] | None = None,
) -> str:
    selected_mode = mode if mode in MODE_PROMPTS else "general"
    config = normalize_rag_config(rag_config)

    if selected_mode == "extract":
        return extract_text(question, config, filters)

    cache_key = build_response_cache_key(question, selected_mode, config, filters)
    cached_response = get_cached_response(cache_key)
    if cached_response is not None:
        logger.info("RAG response cache hit for answer query.")
        return cached_response

    docs = retrieve_documents(
        question=question,
        mode=selected_mode,
        rag_config=config,
        filters=filters,
    )

    if not docs:
        return json.dumps(
            fallback_payload(
                selected_mode,
                (
                    "No encontré fragmentos relevantes en Qdrant Cloud para "
                    "responder esta pregunta con evidencia suficiente."
                ),
            ),
            ensure_ascii=False,
        )

    context = build_context(docs)
    llm = build_llm(num_predict=config["num_predict"])
    chain = prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "context": context,
            "mode": selected_mode,
            "mode_prompt": MODE_PROMPTS[selected_mode],
            "detail_level": config["detail_level"],
        }
    )

    parsed = model_payload(response.content, selected_mode)
    parsed["mode"] = selected_mode
    parsed["sources_used"] = [doc_to_source(doc) for doc in docs]
    parsed["grouped_sources"] = group_sources(docs)
    parsed.setdefault("answer", parsed.get("short_answer", ""))

    if config["debug_chunks"]:
        parsed["debug_chunks"] = debug_chunks(docs)

    response_json = json.dumps(parsed, ensure_ascii=False)
    set_cached_response(cache_key, response_json)
    return response_json
