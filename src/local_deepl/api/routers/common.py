import os

from fastapi.responses import JSONResponse

from local_deepl.api.services.security import SERVER_ERROR_MESSAGE, cleanup_files


def _cleanup(*paths):
    cleanup_files(*paths)


def _stable_server_error(status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": SERVER_ERROR_MESSAGE}
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _path_exists(path: str) -> bool:
    return os.path.exists(path)
