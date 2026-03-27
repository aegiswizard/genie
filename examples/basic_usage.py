"""Genie 🧞‍♀️ — Basic usage example"""
from genie import analyze
from genie.summarizer import format_text_report, format_whatsapp_reply

# Audio-only
result = analyze("sample_audio.wav")
print(format_text_report(result))
print(result.to_agent_json())

# WhatsApp reply format
print(format_whatsapp_reply(result))
