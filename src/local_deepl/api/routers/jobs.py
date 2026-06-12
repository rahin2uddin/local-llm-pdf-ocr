import asyncio

from fastapi import APIRouter

from local_deepl.api.routers import state

router = APIRouter()


@router.get("/api/jobs")
async def get_jobs():
    """Return the recent job history (newest first)."""
    return state.job_history.list()


@router.delete("/api/jobs")
async def clear_jobs():
    """Clear recent job history and current text artifacts."""
    await asyncio.to_thread(state.text_artifacts.clear)
    state.job_history.clear()
    return {"status": "ok"}
