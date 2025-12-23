import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration values for the web app, loaded from environment with sensible defaults."""
    IDLE_TIMEOUT_SECONDS: int = int(os.getenv("IDLE_TIMEOUT_SECONDS", "60"))
    ENABLE_IDLE_SHUTDOWN: bool = bool(os.getenv("FLY_APP_NAME", False))

    TRANSCRIPTION_BUFFER_DURATION_MS: int = int(os.getenv("TRANSCRIPTION_BUFFER_DURATION_MS", "10000"))
    TRANSCRIPTION_BUFFER_OVERLAP_MS: int = int(os.getenv("TRANSCRIPTION_BUFFER_OVERLAP_MS", "2000"))

    # Default expected client chunk interval. Client should send actual chunk in the ws 'start' msg w/ `chunk_interval_ms`
    DEFAULT_CHUNK_INTERVAL_MS: int = int(os.getenv("DEFAULT_CHUNK_INTERVAL_MS", "10000"))


# Single shared config instance
CONFIG = ServerConfig()
