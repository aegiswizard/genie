# Genie 🧞‍♀️ — Licensing

## Genie core

MIT License. See [../LICENSE](../LICENSE).

## Optional dependencies

| Package | License | Use in Genie |
|---------|---------|--------------|
| numpy | BSD-3 | Core acoustic features — always used |
| librosa | ISC | Better audio features — optional |
| DeepFace | MIT | Face emotion analysis |
| Py-FEAT | MIT (model licenses vary) | Rich face analysis — check individual model licenses |
| pyannote.audio | MIT | Speaker diarization — requires free HF account + accepting terms |
| transformers | Apache 2.0 | Speech emotion model |
| matplotlib | PSF | Visual output |
| FastAPI | MIT | REST API |

## Research add-ons

### TRIBE v2
- License: **CC BY-NC 4.0** — noncommercial research use only
- Source: https://github.com/cfosco/TRIBE
- Use: content-level brain response prediction (NOT person-state)
- Do NOT use in commercial products

### brain_language_nlp
- License: MIT
- Source: https://github.com/brain-score/language
- Use: text cognitive analysis

## Py-FEAT model note

Py-FEAT's Python wrapper is MIT, but individual pretrained models may have
their own licenses. Check the Py-FEAT documentation for your chosen model before
deploying in commercial contexts.
