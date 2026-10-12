# Multimodal AI Assistant & VPS Supervisor

A self-learning, multimodal AI assistant powered by **Google Gemini** and deployed via **Telegram**, featuring persistent memory across model upgrades, hardware monitoring, and Docker supervisor capabilities for Oracle Cloud VPS.

---

## 🌟 Key Features

* **Multimodal Intelligence**: Text, High-Resolution Photos, Voice Notes, Video Clips, and Multi-page Documents.
* **Persistent Self-Learning**: SQLite-backed rules and user facts engine. Learns your behavioral preferences and facts automatically. Model-agnostic and durable.
* **Smart 1-Request Filter**: Casual questions strictly use 1 API call; background reflection is triggered only on explicit learning/correction signals.
* **Media Generation**:
  * 🎨 **Google Image Generation**: High-resolution image generation via Google Gemini Image models (`/draw <prompt>`).
  * 🎬 **Google Veo 3.1 Video**: Cinematic AI video generation with progress polling (`/video <prompt>`).
  * 🗣️ **Neural Voice (TTS)**: Realistic bilingual speech synthesis (Vietnamese & English) via Microsoft Neural Voice (`/speak <text>`).
* **VPS & Docker Supervision**:
  * 📊 **Resource Health**: Live CPU, 24GB RAM, Disk, and Uptime metrics (`/vps`).
  * 🐳 **Container Status**: Inspect running Docker containers on host (`/docker`).
  * 📋 **MLTB Logs**: Tail the last $N$ lines of download logs from `mirrorbot-app-1` (`/mltb_logs`).
  * 🔄 **Safe Restart**: Send restart signals to freeze-prone containers (`/mltb_restart`).
* **Automated Background Operations**:
  * ☀️ **Daily Health Briefing**: Automated executive summary sent every morning at 08:00 AM (`/report`).
  * 📦 **Atomic Database Backup**: Scheduled SQLite database snapshot delivered directly to Telegram (`/backup`).
  * 🚨 **Proactive Storage Watchdog**: Automatic alerts when disk capacity exceeds 85%.
* **Group & Control Room Integration**:
  * Zero-spam target filtering: In groups, the bot stays silent unless mentioned (`@bot_username`), replied to, or given commands.

---

## 🏗️ Project Architecture

```
Own AI/
├── bot.py                  # Main Telegram bot interface & command router
├── config.py               # Pydantic Settings schema & validation
├── core/
│   ├── exceptions.py       # Typed exception hierarchy
│   ├── gemini_client.py    # Async multimodal Gemini client with 503 failover
│   ├── media_generator.py  # Google Image, Veo 3.1 Video, & Edge-TTS
│   ├── memory.py           # Thread-safe SQLite persistent memory engine
│   ├── models.py           # Frozen domain models (LearnedRule, VPSMetrics, etc.)
│   ├── reflector.py        # Smart 1-request background reflection agent
│   ├── scheduler.py        # Automated daily reports & SQLite backups
│   ├── telegram_utils.py   # Group mention stripping & authorization
│   └── vps_manager.py      # VPS health metrics & Docker supervisor
├── tests/                  # 73 automated tests (92% coverage)
├── Dockerfile              # Linux ARM64 / AMD64 multi-arch container
├── docker-compose.yml      # Docker Compose mounting ./data & docker.sock
├── requirements.txt        # Production Python dependencies
└── .env.example            # Environment configuration template
```

---

## 🚀 Quick Start (Local Run)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hoangpq14736/Own-AI.git ~/own-ai
   cd ~/own-ai
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   TELEGRAM_BOT_TOKEN=your_bot_token
   ALLOWED_USER_IDS=your_telegram_id_here
   ```

4. **Run the Bot**:
   ```bash
   python bot.py
   ```

5. **Run the Automated Tests**:
   ```bash
   pytest --cov=core --cov=config
   ```

---

## 🐳 Docker Deployment (Oracle Cloud VPS)

```bash
# Start in background with automatic restart
docker compose up -d --build

# View live logs
docker compose logs -f
```

---

## 🔒 Security & Privacy

* **Zero Leaked Secrets**: All API tokens and IDs are stored in `.env` (ignored by Git).
* **Boundary Validation**: Docker container names and prompt inputs are strictly validated against regex whitelists to prevent command injection.
* **Volume Isolation**: SQLite database lives in `./data` which is mounted persistently.
