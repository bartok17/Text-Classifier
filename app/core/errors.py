class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server is unreachable or errors."""


class OllamaBadResponseError(RuntimeError):
    """Raised when Ollama responds but the content is unusable/invalid."""
