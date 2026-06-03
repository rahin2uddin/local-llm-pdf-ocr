from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from pdf_ocr.api.services.progress import ProgressService

router = APIRouter()
_progress_service = ProgressService()


class ConnectionManager:
    """Tracks token-bound WebSocket progress channels."""

    def __init__(self) -> None:
        self.active: dict[str, WebSocket] = {}
        self._tokens: dict[str, str] = {}

    async def connect(
        self, websocket: WebSocket, channel_id: str, session_token: str
    ) -> None:
        channel_id = _progress_service.validate_channel_id(channel_id)
        session_token = _progress_service.validate_session_token(session_token)
        await websocket.accept()
        self.active[channel_id] = websocket
        self._tokens[channel_id] = session_token

    def disconnect(self, channel_id: str) -> None:
        self.active.pop(channel_id, None)
        self._tokens.pop(channel_id, None)

    def is_authorized(self, channel_id: str | None, session_token: str | None) -> bool:
        if not channel_id or not session_token:
            return False
        expected_token = self._tokens.get(channel_id)
        if expected_token is None:
            return False
        try:
            return _progress_service.is_bound(
                channel_id=channel_id,
                session_token=session_token,
                expected_channel_id=channel_id,
                expected_session_token=expected_token,
            )
        except (TypeError, ValueError):
            return False

    async def send_progress(
        self, channel_id: str | None, message: str, percent: int, stage: str = ""
    ) -> None:
        """Send a JSON progress frame to an active authorized channel."""
        if not channel_id:
            return
        ws = self.active.get(channel_id)
        if ws is None:
            return
        try:
            await ws.send_json({"status": message, "percent": percent, "stage": stage})
        except Exception:
            self.disconnect(channel_id)


manager = ConnectionManager()


@router.post("/api/progress/session")
async def create_progress_session(body: dict | None = None):
    """Issue an opaque websocket progress channel and binding token."""
    display_client_id = body.get("client_id") if body else None
    try:
        channel = _progress_service.create_channel(display_client_id=display_client_id)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid progress session parameters."},
        )
    return {
        "channel_id": channel.channel_id,
        "session_token": channel.session_token,
    }


@router.websocket("/ws/{channel_id}")
async def websocket_endpoint(websocket: WebSocket, channel_id: str, token: str = ""):
    """Accept a token-bound WebSocket connection for real-time progress updates."""
    try:
        await manager.connect(websocket, channel_id, token)
    except (TypeError, ValueError):
        await websocket.close(code=1008)
        return
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(channel_id)
