"""Lightweight, file-backed user authentication.

Stores users in ``analysis/data/users.json`` with PBKDF2-HMAC-SHA256 hashed
passwords (standard library only — no extra dependencies). A default account is
seeded on first run so the login screen works out of the box:

    username: admin
    password: admin123   (override with the APP_PASSWORD env var)

Change the default credentials in production, or register a new user from the
login screen and delete the default.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_USERS_PATH = os.path.join(_DATA_DIR, "users.json")
_PBKDF2_ROUNDS = 200_000
_lock = threading.Lock()


def _hash_password(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS
    )
    return dk.hex()


def _load() -> dict:
    if not os.path.exists(_USERS_PATH):
        return {}
    try:
        with open(_USERS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _save(users: dict) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp = _USERS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)
    os.replace(tmp, _USERS_PATH)


def _norm(username: str) -> str:
    return (username or "").strip().lower()


def user_exists(username: str) -> bool:
    return _norm(username) in _load()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Create a new user. Returns (ok, message)."""
    username = _norm(username)
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    with _lock:
        users = _load()
        if username in users:
            return False, "That username is already taken."
        salt = secrets.token_hex(16)
        users[username] = {"salt": salt, "hash": _hash_password(password, salt)}
        _save(users)
    return True, "Account created."


def verify(username: str, password: str) -> bool:
    """Return True if the username/password pair is valid."""
    username = _norm(username)
    user = _load().get(username)
    if not user:
        return False
    expected = user.get("hash", "")
    actual = _hash_password(password or "", user.get("salt", ""))
    return hmac.compare_digest(expected, actual)


def ensure_default_user() -> None:
    """Seed a default admin account on first run if no users exist."""
    with _lock:
        users = _load()
        if users:
            return
        username = _norm(os.environ.get("APP_USERNAME", "admin"))
        password = os.environ.get("APP_PASSWORD", "admin123")
        salt = secrets.token_hex(16)
        users[username] = {"salt": salt, "hash": _hash_password(password, salt)}
        _save(users)
