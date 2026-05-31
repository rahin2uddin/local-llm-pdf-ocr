import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from pdf_ocr.api.routers.config import _config

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class TranslationState(TypedDict):
    source_chunk: str
    target_language: str
    rag_context: list[str]
    translated_chunk: str
    evaluation_score: float
    feedback: str
    attempts: int

# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
db_client = None
emb_fn = None

def get_chroma_collection():
    if not HAS_CHROMA:
        return None

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
        return db_client.get_collection(name="lanes_lexicon", embedding_function=emb_fn)  # type: ignore[arg-type]
    except Exception:
        return None

def retrieve_lexicon_context(state: TranslationState):
    """Retrieves terminology from ChromaDB."""
    collection = get_chroma_collection()
    context = []

    if collection:
        try:
            results = collection.query(
                query_texts=[state["source_chunk"]],
                n_results=3
            )
            if results and results.get("documents") and results["documents"][0]:
                context = results["documents"][0]
        except Exception:
            pass

    return {"rag_context": context}

def translate_node(state: TranslationState):
    """Calls the LLM to translate the chunk, using RAG context."""
    import litellm
    active_api_base = _config.get("api_base", "http://localhost:1234/v1")
    active_api_key = _config.get("api_key", "lm-studio")
    active_model = _config.get("model", "gemma-4-e4b-it-obliterated")

    from pdf_ocr.utils.litellm_provider import resolve_custom_provider
    custom_provider = resolve_custom_provider(active_model)

    prompt = f"Translate the following text into {state['target_language']}.\n\n"
    if state.get("rag_context"):
        prompt += "Use the following lexicon definitions to ensure correct terminology:\n"
        prompt += "\n".join(state["rag_context"]) + "\n\n"

    if state.get("feedback"):
        prompt += f"Previous translation had issues. Feedback: {state['feedback']}\nPlease fix these issues.\n\n"

    prompt += f"SOURCE TEXT:\n{state['source_chunk']}"

    try:
        response = litellm.completion(
            model=active_model,
            custom_llm_provider=custom_provider,
            api_base=active_api_base,
            api_key=active_api_key,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        translated = response.choices[0].message.content or ""
    except Exception as e:
        translated = f"[Translation Error: {e}]"

    return {"translated_chunk": translated, "attempts": state.get("attempts", 0) + 1}

def evaluate_node(state: TranslationState):
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

    if len(translated) < len(state["source_chunk"]) * 0.1:
        return {"evaluation_score": 0.0, "feedback": "Translation too short. Ensure you translate the entire chunk."}

    return {"evaluation_score": 1.0, "feedback": "Looks good"}

def should_refine(state: TranslationState):
    """Router logic for conditional edge."""
    if state.get("evaluation_score", 1.0) < 0.8:
        return "translate"
    return "end"

# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------
workflow = StateGraph(TranslationState)
workflow.add_node("retrieve", retrieve_lexicon_context)
workflow.add_node("translate", translate_node)
workflow.add_node("evaluate", evaluate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "translate")
workflow.add_edge("translate", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    should_refine,
    {
        "translate": "translate",
        "end": END
    }
)

translation_app = workflow.compile()

def run_translation(text: str, target_language: str = "English") -> str:
    """Convenience function to run the compiled graph on a single chunk."""
    initial_state = {
        "source_chunk": text,
        "target_language": target_language,
        "rag_context": [],
        "translated_chunk": "",
        "evaluation_score": 1.0,
        "feedback": "",
        "attempts": 0
    }

    result = translation_app.invoke(initial_state)  # type: ignore[call-overload]
    return result.get("translated_chunk", "")
