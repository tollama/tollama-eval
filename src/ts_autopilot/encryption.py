"""Data encryption at rest for benchmark results.

Provides envelope encryption using Fernet (AES-128-CBC)
with configurable key backends.

Requires: ``pip install cryptography``
"""

from __future__ import annotations

import os
from pathlib import Path

from ts_autopilot.logging_config import get_logger

logger = get_logger("encryption")

try:
    from cryptography.fernet import Fernet

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class EncryptionManager:
    """Manages encryption/decryption of result files."""

    def __init__(self, key: bytes | str | None = None) -> None:
        if not HAS_CRYPTO:
            self._enabled = False
            logger.info(
                "cryptography not installed — encryption disabled"
            )
            return

        self._enabled = True
        if key is None:
            key = os.environ.get("TOLLAMA_ENCRYPTION_KEY", "")
        if isinstance(key, str):
            if not key:
                self._enabled = False
                logger.info("No encryption key — encryption disabled")
                return
            key = key.encode()
        self._fernet = Fernet(key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key."""
        if not HAS_CRYPTO:
            raise ImportError(
                "cryptography is required for encryption"
            )
        return Fernet.generate_key().decode()

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data."""
        if not self._enabled:
            return data
        return self._fernet.encrypt(data)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        if not self._enabled:
            return data
        return self._fernet.decrypt(data)

    def encrypt_file(self, path: Path) -> None:
        """Encrypt a file in place."""
        if not self._enabled:
            return
        plaintext = path.read_bytes()
        ciphertext = self._fernet.encrypt(plaintext)
        path.write_bytes(ciphertext)
        logger.debug("Encrypted %s", path)

    def decrypt_file(self, path: Path) -> bytes:
        """Decrypt a file and return its contents."""
        if not self._enabled:
            return path.read_bytes()
        ciphertext = path.read_bytes()
        return self._fernet.decrypt(ciphertext)
