# memory.md — Self-Improving Session Memory

This file is updated by the agent when it learns new facts, corrects errors, or discovers user preferences.
The agent reads this file at the start of every session.

## Confirmed Data Sources

- **USGS Site 06730200** = Boulder Creek at North 75th Street, Colorado. Used for the 2013 Colorado Flood simulation. Danger threshold: 10.0 ft gage height.
- **CrisisLex dataset path**: `data/text/CrisisLexT26/CrisisLexT26/2013_Colorado_floods/2013_Colorado_floods-tweets_labeled.csv`
- **Open-Meteo coordinates for Boulder, CO**: latitude=40.0150, longitude=-105.2705
- **xView2 archive**: `data/imagery/test_images_labels_targets.tar` (2.7 GB, stream-extracted, do not unzip fully)

## Confirmed System Specs

- Groq model: `llama-3.3-70b-versatile` (not llama3-70b-8192 — that was decommissioned)
- Vision model: Ollama LLaVA 7B at `http://localhost:11434`
- Embeddings: `BAAI/bge-small-en-v1.5` via HuggingFaceEmbeddings

## Known Issues Fixed

- 2026-02: `langchain.chains` was removed in newer langchain — replaced with custom SimpleRAGChain
- 2026-02: `llama3-70b-8192` decommissioned by Groq — switched to `llama-3.3-70b-versatile`
- 2026-04: `HuggingFaceBgeEmbeddings` from langchain-community is deprecated — use `langchain_huggingface`
- 2026-04: `Chroma` from langchain-community is deprecated — use `langchain_chroma`

## User Preferences

- Always production-grade code — no mock/synthetic data in ingestion
- IEEE-publishable results — requires real data backing all claims
- Free infrastructure only — no paid APIs
- Minimize disk usage — stream from archives rather than extracting fully
