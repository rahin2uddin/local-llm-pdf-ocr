from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    """Tracks active WebSocket connections and dispatches progress updates."""

    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active.pop(client_id, None)

    async def send_progress(
        self, client_id: str, message: str, percent: int, stage: str = ""
    ):
        """Send a JSON progress frame including the pipeline stage name."""
        ws = self.active.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json(
                {"status": message, "percent": percent, "stage": stage}
            )
        except Exception:
            self.disconnect(client_id)


manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Accept a WebSocket connection for real-time progress updates."""
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

