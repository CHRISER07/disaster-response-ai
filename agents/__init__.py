# agents/__init__.py
# Windows bare-metal compatibility guard: protect against torch WinError 193 DLL crash
import sys
try:
    import torch
except (OSError, ImportError):
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = types.SimpleNamespace(origin="fake", submodule_search_locations=None)
    fake_torch.__version__ = "2.2.0"
    sys.modules["torch"] = fake_torch

from agents.disaster_agent import run_agent, get_agent

__all__ = ["run_agent", "get_agent"]
