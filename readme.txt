#From your project folder
python -m venv .venv
.\.venv\Scripts\Activate

#pip and install required packages
python -m pip install --upgrade pip wheel
pip install aiohttp trafilatura beautifulsoup4 lxml tldextract

#for JS-rendered pages
pip install playwright
python -m playwright install


#to run and download
python .\site_dump.py https://www.instagram.com/i_m_abhipatel/?igsh=MXI5YjFqazRkZDkxaA%3D%3D# `
  --out-dir .\crawl `
  --include-subdomains `
  --ignore-robots `
  --concurrency 2 `
  --max-pages 200 `
  --delay-ms 250 `
  --dataset `
  --include-pdfs `
  --export-urls `
  --verbose
