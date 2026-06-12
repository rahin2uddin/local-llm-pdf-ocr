from local_deepl.api.services.artifacts import TextArtifactStore
from local_deepl.api.services.jobs import JobHistory
from local_deepl.api.services.progress import ProgressService

text_artifacts = TextArtifactStore()
metadata_artifacts = TextArtifactStore()
export_artifacts = TextArtifactStore()
job_history = JobHistory()
progress_service = ProgressService()
