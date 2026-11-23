"""Centralized exception hierarchy for the inference optimizer."""


class InferenceOptimizerError(Exception):
    """Base class for all project-specific errors."""


class HardwareDetectionError(InferenceOptimizerError):
    """Raised when hardware probing fails in an unrecoverable way."""


class ConfigurationError(InferenceOptimizerError):
    """Raised when application configuration is invalid or missing."""


class ModelLoadError(InferenceOptimizerError):
    """Raised when a model fails to load (missing file, bad GGUF, etc.)."""


class BackendStartupError(InferenceOptimizerError):
    """Raised when a backend server process fails to start or never becomes ready."""


class BackendConnectionError(InferenceOptimizerError):
    """Raised when a running backend can't be reached or returns a bad response."""


class UnsupportedHardwareError(InferenceOptimizerError):
    """Raised when a backend can't run on the current OS/GPU combination."""
