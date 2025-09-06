# Veritas Scraper

Full-site crawler with a React + Tailwind UI and a Node/Express API that runs `site_dump.py`. Saves outputs under `crawl/<jobId>/` and zips the run for download.

## Contents (this repo)

* `site_dump.py` – Python crawler.
* `requirements.txt` – Python deps.
* `backend.zip` – API server (Express).
* `frontend.zip` – Web UI (Vite + React + Tailwind).
* `LICENSE`, `README.md`.

> Outputs are created at `crawl/<jobId>/`.

---

## Prerequisites

* Python 3.10+
* Node.js 18+ and npm
* Git (optional)

---

## Unzip layout

Create two folders at the repo root named `backend/` and `frontend/`, then extract the zips into them.

**Windows (PowerShell)**

```powershell
Expand-Archive -Force .\backend.zip -DestinationPath .\backend
Expand-Archive -Force .\frontend.zip -DestinationPath .\frontend
```

**macOS/Linux**

```bash
unzip -o backend.zip -d backend
unzip -o frontend.zip -d frontend
```

You should now have:

```
.
├─ site_dump.py
├─ requirements.txt
├─ backend/
└─ frontend/
```

---

## Install

### 1) Python env

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install
```

### 2) Backend API

```powershell
cd backend
copy .env.example .env
npm install
npm run dev
```

* The server listens on `http://localhost:5001`.
* Set the Python path in `backend/.env` if needed:

`backend/.env`

```
PORT=5001
CLIENT_ORIGIN=http://localhost:5173
PYTHON_EXE=C:\full\path\to\.venv\Scripts\python.exe   # Windows
PROJECT_ROOT=..                                       # repo root
```

macOS/Linux example:

```
PYTHON_EXE=/full/path/to/.venv/bin/python
```

### 3) Frontend UI

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Use

1. Go to `http://localhost:5173`.
2. Enter a start URL.
3. Configure options (subdomains, PDFs, JS render, max pages, etc.).
4. Click **Run crawl**.
5. When complete, click **Download ZIP**.

Output structure:

```
crawl/<jobId>/
  00001-*/page.html
  00001-*/page.json
  _assets/img/*
  _dataset/pages.jsonl
  _dataset/text.jsonl
  _dataset/links.jsonl
  graph.json
  index.jsonl
  crawl.log
```

ZIP: `crawl/<jobId>.zip`.

---

## API (optional)

Start a crawl:

```bash
curl -X POST http://localhost:5001/api/crawl \
  -H "content-type: application/json" \
  -d '{"url":"https://example.com","maxPages":50,"concurrency":2,"delayMs":250,"includeSubdomains":true,"ignoreRobots":true,"includePdfs":true,"dataset":true,"renderAll":false}'
```

* Download: `GET /api/crawl/:jobId/download`
* Run log: `GET /api/crawl/:jobId/log`

> `GET /` on the backend will print “Cannot GET /” because it’s an API. Use the UI on port 5173.

---

## Branding (logo)

Place a logo at `frontend/public/logo.png` and replace the square in `App.tsx`:

```tsx
<img src="/logo.png" alt="Veritas Scraper" className="h-9 w-9 rounded-xl object-cover" />
```

---

## Git push (optional)

Initialize and push:

```bash
git init
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File backend\scripts\git-push.ps1 -remote origin -branch main -message "feat: initial"
```

macOS/Linux:

```bash
bash backend/scripts/git-push.sh origin main "feat: initial"
```

---

## Troubleshooting

* **CORS error**: set `CLIENT_ORIGIN` in `backend/.env` to `http://localhost:5173` and restart the backend.
* **Python not found**: point `PYTHON_EXE` to your venv python.
* **Playwright error**: run `python -m playwright install` in the venv.
* **Crawl failed**: check `crawl/<jobId>/server-run.log`.
* **Empty output**: increase `--max-pages`, enable `--include-subdomains`, and consider `Render JS`.
* **Permissions** (Windows PowerShell scripts):
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Bypass`.

---

## License

MIT (or your chosen license).
