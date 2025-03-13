class PyProsodyBaseException(Exception):
    """Base exception for all PyProsody errors."""
    pass

class ConfigurationError(PyProsodyBaseException):
    """Raised when there's an error in configuration."""
    pass

class InputValidationError(PyProsodyBaseException):
    """Raised when input validation fails."""
    pass

class ProcessingError(PyProsodyBaseException):
    """Base class for processing-related errors."""
    pass

class TextProcessingError(ProcessingError):
    """Raised when text processing fails."""
    pass

class EmotionAnalysisError(ProcessingError):
    """Raised when emotion analysis fails."""
    pass

class AudioGenerationError(ProcessingError):
    """Raised when audio generation fails."""
    pass

class ResourceError(PyProsodyBaseException):
    """Base class for resource-related errors."""
    pass

class ModelLoadError(ResourceError):
    """Raised when model loading fails."""
    pass

class PipelineError(PyProsodyBaseException):
    """Raised when pipeline orchestration fails."""
    pass