"""FastAPI service wrapping an InferenceBackend. The backend stays framework-agnostic;
this package only adds HTTP plumbing (schemas, routing, concurrency admission, metrics).
"""

from inference_optimizer.api.app import create_app

__all__ = ["create_app"]
