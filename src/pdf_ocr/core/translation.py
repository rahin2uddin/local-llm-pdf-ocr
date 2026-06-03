from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

from pdf_ocr.core.translation_config import (
    AsyncTranslationUnavailable,
    TranslationSettings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class TranslationState(TypedDict, total=False):
    source_chunk: str
    target_language: str
    rag_context: list[str]
    translated_chunk: str
    evaluation_score: float
    feedback: str
    attempts: int
    settings: TranslationSettings


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
db_client: Any | None = None
emb_fn: Any | None = None
_translation_app: Any | None = None


def _optional_dependency_message(package: str) -> str:
    return (
        f"Async translation requires optional dependency '{package}'. "
        "Install the async translation extras to enable this feature."
    )


def _get_chroma_modules() -> tuple[Any, Any] | None:
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        return None
    return chromadb, embedding_functions


def get_chroma_collection() -> Any | None:
    modules = _get_chroma_modules()
    if modules is None:
        return None
    chromadb, embedding_functions = modules

    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "chroma_db")
    if not os.path.exists(db_path):
        return None

    global db_client, emb_fn
    try:
        if db_client is None:
            db_client = chromadb.PersistentClient(path=db_path)
        if emb_fn is None:
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
        return db_client.get_collection(name="lanes_lexicon", embedding_function=emb_fn)
    except Exception as exc:
        logger.warning("Unable to load translation lexicon from ChromaDB: %s", exc)
        return None


def retrieve_lexicon_context(state: TranslationState) -> dict[str, list[str]]:
    """Retrieves terminology from ChromaDB."""
    collection = get_chroma_collection()
    context: list[str] = []

    if collection:
        try:
            results = collection.query(query_texts=[state["source_chunk"]], n_results=3)
            if results and results.get("documents") and results["documents"][0]:
                context = results["documents"][0]
        except Exception as exc:
            logger.warning("Unable to retrieve translation lexicon context: %s", exc)

    return {"rag_context": context}


def _state_settings(state: TranslationState) -> TranslationSettings:
    settings = state.get("settings")
    if settings is None:
        return TranslationSettings.from_env()
    if not isinstance(settings, TranslationSettings):
        raise ValueError("translation state settings must be TranslationSettings")
    return settings


def translate_node(state: TranslationState) -> dict[str, str | int]:
    """Calls the LLM to translate the chunk, using RAG context."""
    import litellm

    from pdf_ocr.utils.litellm_provider import resolve_custom_provider

    settings = _state_settings(state)
    custom_provider = resolve_custom_provider(settings.model)

    prompt = f"Translate the following text into {state['target_language']}.\n\n"
    if state.get("rag_context"):
        prompt += (
            "Use the following lexicon definitions to ensure correct terminology:\n"
        )
        prompt += "\n".join(state["rag_context"]) + "\n\n"

    if state.get("feedback"):
        prompt += f"Previous translation had issues. Feedback: {state['feedback']}\nPlease fix these issues.\n\n"

    prompt += f"SOURCE TEXT:\n{state['source_chunk']}"

    try:
        response = litellm.completion(
            model=settings.model,
            custom_llm_provider=custom_provider,
            api_base=settings.api_base,
            api_key=settings.api_key,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        translated = response.choices[0].message.content or ""
    except Exception as e:
        translated = f"[Translation Error: {e}]"

    return {"translated_chunk": translated, "attempts": state.get("attempts", 0) + 1}


def evaluate_node(state: TranslationState) -> dict[str, float | str]:
    """Evaluates the translation quality."""
    # Simplified mock evaluator.
    # In a full production system, this would be another LLM call checking if glossary terms were used.
    attempts = state.get("attempts", 0)

    translated = state.get("translated_chunk", "")
    if translated.startswith("[Translation Error"):
        if attempts >= 3:
            return {"evaluation_score": 1.0, "feedback": "Failed after max attempts."}
        return {"evaluation_score": 0.0, "feedback": "Translation API call failed."}

    if attempts >= 3:
        # Force accept after 3 tries to prevent infinite loops
        return {"evaluation_score": 1.0, "feedback": ""}

    # If the source chunk has no letters or is extremely short, skip length ratio check
    source = state.get("source_chunk", "")
    has_letters = any(c.isalpha() for c in source)
    if not has_letters or len(source.strip()) < 5:
        return {"evaluation_score": 1.0, "feedback": "Looks good"}

    if len(translated) < len(source) * 0.1:
        return {
            "evaluation_score": 0.0,
            "feedback": "Translation too short. Ensure you translate the entire chunk.",
        }

    return {"evaluation_score": 1.0, "feedback": "Looks good"}


def should_refine(state: TranslationState) -> str:
    """Router logic for conditional edge."""
    if state.get("evaluation_score", 1.0) < 0.8:
        return "translate"
    return "end"


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------


def get_translation_app() -> Any:
    """Return the compiled LangGraph app, building it only when invoked."""
    global _translation_app
    if _translation_app is not None:
        return _translation_app

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise AsyncTranslationUnavailable(
            _optional_dependency_message("langgraph")
        ) from exc

    workflow = StateGraph(TranslationState)
    workflow.add_node("retrieve", retrieve_lexicon_context)
    workflow.add_node("translate", translate_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "translate")
    workflow.add_edge("translate", "evaluate")
    workflow.add_conditional_edges(
        "evaluate", should_refine, {"translate": "translate", "end": END}
    )

    _translation_app = workflow.compile()
    return _translation_app


class _LazyTranslationApp:
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return get_translation_app().invoke(*args, **kwargs)


translation_app = _LazyTranslationApp()


def chunk_text(text: str, max_chunk_size: int = 4000) -> list[str]:
    """Splits text into chunks of maximum size, trying to preserve paragraph and sentence boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be greater than zero")
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk: list[str] = []
    current_len = 0

    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if len(para) + 2 > max_chunk_size:
            # Paragraph itself is too large, split by lines
            lines = para.split("\n")
            for line in lines:
                if len(line) + 1 > max_chunk_size:
                    # Split by words
                    words = line.split(" ")
                    for word in words:
                        if current_len + len(word) + 1 > max_chunk_size:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            current_chunk = [word]
                            current_len = len(word)
                        else:
                            current_chunk.append(word)
                            current_len += len(word) + 1
                else:
                    if current_len + len(line) + 1 > max_chunk_size:
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                        current_chunk = [line]
                        current_len = len(line)
                    else:
                        current_chunk.append(line)
                        current_len += len(line) + 1
        else:
            if current_len + len(para) + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para) + 2

    if current_chunk:
        chunks.append(("\n\n" if "\n\n" in text else "\n").join(current_chunk))

    return [c for c in chunks if c.strip()]


def run_translation(
    text: str,
    target_language: str = "English",
    settings: TranslationSettings | None = None,
) -> str:
    """Convenience function to run the compiled graph on a text by chunking it to prevent LLM context overflow."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("target_language must be a non-empty string")
    if not text.strip():
        return ""

    active_settings = settings or TranslationSettings.from_env()
    chunks = chunk_text(text)
    translated_chunks: list[str] = []
    app = get_translation_app()

    for chunk in chunks:
        initial_state: TranslationState = {
            "source_chunk": chunk,
            "target_language": target_language,
            "rag_context": [],
            "translated_chunk": "",
            "evaluation_score": 1.0,
            "feedback": "",
            "attempts": 0,
            "settings": active_settings,
        }
        result = app.invoke(initial_state)
        translated = result.get("translated_chunk", "")
        if translated:
            translated_chunks.append(translated)

    return "\n\n".join(translated_chunks)
