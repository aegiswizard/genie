# Genie 🧞‍♀️ — Integrations

## OpenClaw / Hermes / Any agent
Drop `skill.md` into your agent's skills folder.

## Python
```python
from genie import analyze
result = analyze("file.m4a")
print(result.to_agent_json())
```

## REST API
```bash
genie serve --port 7070
curl -X POST http://localhost:7070/analyze -F "file=@voice.m4a"
```

## WhatsApp bot
1. Receive media file from WhatsApp API
2. Save to temp path
3. `result = analyze(path)`
4. Reply with `format_whatsapp_reply(result)`

## CLI pipeline
```bash
genie analyze voice.m4a --output json | jq .primary_state
```
