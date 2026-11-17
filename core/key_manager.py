"""
core/key_manager.py

Thread-safe, multi-account round-robin key manager with automatic failover.
Supports both Gemini (Google AI Studio) and Groq account pools.
Automatically tracks and disables keys that return 403 PERMISSION_DENIED or invalid auth.
"""
import os
import threading
from typing import List, Optional, Set
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
# Explicit shell/CI environment variables take precedence over .env defaults.
load_dotenv(_env_path, override=False)
load_dotenv(override=False)


class KeyPool:
    """Thread-safe round-robin key pool with health tracking and automatic exclusion."""
    def __init__(self, key_name_prefix: str, legacy_fallback: str):
        self.lock = threading.Lock()
        self.keys: List[str] = []
        self.bad_keys: Set[str] = set()
        
        # Prioritize fresh keys first to maximize fresh quota
        if key_name_prefix == "GROQ_API_KEY":
            order = ['J', 'K', 'L', 'H', 'I', 'E', 'F', 'G', 'A', 'B', 'C', 'D']
        else:
            order = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
        
        for letter in order:
            k = os.getenv(f"{key_name_prefix}_{letter}", "").strip()
            if k and k not in self.keys:
                self.keys.append(k)
                
        # If none of A-D found, fallback to single legacy key
        if not self.keys:
            legacy = os.getenv(legacy_fallback, "").strip()
            if legacy:
                self.keys.append(legacy)
                
        self.index = 0
        self.name = key_name_prefix

    def mark_bad(self, key: str, reason: str = ""):
        """Mark a key as unusable (e.g. 403 or quota exceeded)."""
        with self.lock:
            self.bad_keys.add(key)
            if key in self.keys:
                self.keys.remove(key)
            if self.keys:
                self.index = self.index % len(self.keys)
            print(f"[KeyPool {self.name}] Disabled bad key (...{key[-6:]}): {reason}. Remaining valid: {len(self.keys)}")

    def get_key(self) -> str:
        """Get the next valid key in round-robin order."""
        with self.lock:
            valid_keys = [k for k in self.keys if k not in self.bad_keys]
            if not valid_keys:
                raise ValueError(f"No valid API keys remaining for {self.name}!")
            self.index = self.index % len(valid_keys)
            key = valid_keys[self.index]
            self.index = (self.index + 1) % len(valid_keys)
            return key

    def total_keys(self) -> int:
        with self.lock:
            return len([k for k in self.keys if k not in self.bad_keys])


# Global instances
gemini_pool = KeyPool("GEMINI_API_KEY", "GEMINI_API_KEY")
groq_pool   = KeyPool("GROQ_API_KEY", "GROQ_API_KEY")

# Ensure default environment variables are populated for libraries expecting them
if not os.getenv("GROQ_API_KEY"):
    try:
        os.environ["GROQ_API_KEY"] = groq_pool.get_key()
    except Exception:
        pass

if not os.getenv("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = gemini_pool.get_key()
    except Exception:
        pass


def get_gemini_client():
    """Returns a google.genai client using a rotated key."""
    from google import genai
    key = gemini_pool.get_key()
    return genai.Client(api_key=key), key


def get_groq_client():
    """Returns a Groq client using a rotated key."""
    from groq import Groq
    key = groq_pool.get_key()
    return Groq(api_key=key), key


def get_groq_chat(model: str = "qwen/qwen3.8-27b", temperature: float = 0.1, max_retries: int = 0):
    """Returns a ChatGroq instance with a round-robin key."""
    from langchain_groq import ChatGroq
    key = groq_pool.get_key()
    return ChatGroq(model=model, api_key=key, temperature=temperature, max_retries=max_retries)


def get_gemini_chat(model: str = "gemini-3.1-flash-lite", temperature: float = 0.1, max_retries: int = 2):
    """Returns a ChatGoogleGenerativeAI instance with a round-robin key."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = gemini_pool.get_key()
    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature, max_retries=max_retries)
