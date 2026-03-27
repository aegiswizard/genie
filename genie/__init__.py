"""
Genie 🧞‍♀️ — Agent-native state sketch engine.
Made with ❤️ by Aegis Wizard 🧙‍♂️
MIT License | github.com/aegiswizard/genie

Genie reads the room, not your soul.

Quick start:
    from genie import analyze
    result = analyze("voice_note.m4a")
    print(result.primary_state)
    print(result.to_agent_json())
"""

__version__ = "1.0.0"
__author__  = "Aegis Wizard"
__license__ = "MIT"
__url__     = "https://github.com/aegiswizard/genie"

from .pipeline import run as analyze
from .schemas  import GenieResult, STATE_LABELS
from .config   import GenieConfig

__all__ = ["analyze", "GenieResult", "GenieConfig", "STATE_LABELS"]
