# 🌐 News Intelligence System

AI-powered daily digest pipeline that aggregates, clusters, and summarizes global news from 18+ RSS feeds across Cybersecurity, Technology, and Cryptocurrency — entirely on local hardware with zero cloud dependency.

## Features

- **Semantic deduplication** — Groups similar stories using sentence embeddings (not keyword matching)
- **LLM-powered ranking** — Scores importance using Llama 3.2 3B with multi-signal composite scoring
- **Smart summarization** — Generates concise 3-4 sentence summaries with Llama 3.1 8B
- **Automated delivery** — Sends daily digest to Telegram and email at midnight
- **Fully local** — No API costs, no cloud services, runs on a single RTX 3090

## Tech Stack

- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **LLM Inference:** Ollama (Llama 3.2 3B + Llama 3.1 8B, Q4_K_M quantization)
- **Clustering:** scikit-learn (Agglomerative with cosine similarity)
- **Scheduling:** systemd timer
- **Deployment:** Docker + Proxmox VM with GPU passthrough

## Performance

- **Full pipeline:** ~8-10 minutes (fetch → cluster → rank → summarize → deliver)
- **VRAM usage:** ~8.5 GB (both models loaded)
- **Output:** 12-18 curated stories per digest

## Documentation

See `News_Intelligence_System_Documentation.docx` for complete architecture, bug fixes, and deployment guide.

## Quick Start

1. Clone and install dependencies:
```bash
   git clone https://github.com/YOUR_USERNAME/news-intelligence-system.git
   cd news-intelligence-system
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

2. Configure RSS feeds in `config/sources.yaml`

3. Set up `.env` with your credentials:
```bash
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   SMTP_HOST=smtp.gmail.com
   SMTP_USER=your_email
   SMTP_PASSWORD=your_app_password
   EMAIL_FROM=your_email
   EMAIL_TO=recipient_email
```

4. Install Ollama and pull models:
```bash
   docker run -d --gpus all --name ollama -p 11434:11434 \
     -v ollama-models:/root/.ollama ollama/ollama
   docker exec ollama ollama pull llama3.2:3b-instruct-q4_K_M
   docker exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
```

5. Test run:
```bash
   python advanced_main.py
```

6. Set up automation (optional):
```bash
   sudo cp news-digest.service news-digest.timer /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now news-digest.timer
```

## License

MIT
