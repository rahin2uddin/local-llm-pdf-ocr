Set objShell = CreateObject("WScript.Shell")
scriptDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

' Change working directory
objShell.CurrentDirectory = scriptDir

' 1. Start Redis
objShell.Run "cmd.exe /c docker run -d --name redis-local-ocr -p 6379:6379 redis", 0, True

' 2. Start Celery Worker (completely hidden)
objShell.Run "cmd.exe /c uv run --extra web celery -A src.local_deepl.api.celery_app worker --loglevel=info -P solo", 0, False

' 3. Start Web Server (completely hidden)
objShell.Run "cmd.exe /c uv run --extra web uvicorn src.local_deepl.server:app --port 8000", 0, False

' Wait for services to initialize
WScript.Sleep 5000

' 4. Open Web UI
objShell.Run "http://localhost:8000"
