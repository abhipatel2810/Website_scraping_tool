#!/usr/bin/env python3
import argparse, asyncio, gzip, json, re, time, sys, hashlib, os, traceback
from pathlib import Path
from typing import Optional, Tuple, Set, Dict, List
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
import aiohttp
from aiohttp import ClientTimeout, ClientConnectorError
from bs4 import BeautifulSoup
import tldextract
import trafilatura
import urllib.robotparser as robotparser

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None  # optional

# ---------------- Constants ----------------
SKIP_EXT = re.compile(
    r".*\.(?:jpg|jpeg|png|gif|webp|svg|ico|css|js|mp4|m4v|mov|avi|mp3|wav|zip|gz|tar|rar|7z|woff2?|ttf|eot|pdf|docx?|pptx?|xlsx?)$",
    re.I,
)

# ---------------- Utils ----------------
def canonicalize(u: str) -> str:
    p = urlparse(u)
    query = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True)))
    p = p._replace(scheme=(p.scheme.lower() or "https"),
                   netloc=p.netloc.lower(),
                   fragment="",
                   query=query)
    if (p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443):
        p = p._replace(netloc=p.hostname or "")
    return urlunparse(p)

def same_site(seed: str, target: str, include_subdomains: bool) -> bool:
    s = tldextract.extract(seed)
    t = tldextract.extract(target)
    if include_subdomains:
        return (s.domain, s.suffix) == (t.domain, t.suffix)
    return (s.subdomain, s.domain, s.suffix) == (t.subdomain, t.domain, t.suffix)

def slugify_from_url(u: str, max_len: int = 40) -> str:
    p = urlparse(u)
    path = p.path.strip("/")
    slug = path.split("/")[-1] or "home"
    slug = re.sub(r"[^A-Za-z0-9\-]+", "-", slug).strip("-").lower()
    if not slug:
        slug = "page"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def guess_ext(url: str, ct: str, default: str = "bin") -> str:
    if ct.startswith("image/"):
        m = ct.split("/", 1)[1].split(";")[0].strip()
        return {"jpeg": "jpg"}.get(m, m)
    m = re.search(r"\.(\w{2,5})(?:$|\?)", url)
    if m:
        return m.group(1).lower()
    return default

def clean_text(t: str) -> str:
    # normalize whitespace for model-ready text
    t = re.sub(r'\r\n?', '\n', t)
    t = re.sub(r'[ \t]+\n', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

# ---------------- Robots + Sitemap ----------------
async def load_robots(session: aiohttp.ClientSession, base: str) -> robotparser.RobotFileParser:
    rp = robotparser.RobotFileParser()
    robots_url = urljoin(base, "/robots.txt")
    try:
        async with session.get(robots_url, timeout=ClientTimeout(total=15)) as r:
            if r.status == 200:
                rp.parse((await r.text()).splitlines())
            else:
                rp.disallow_all = False
    except Exception:
        rp.disallow_all = False
    return rp

async def discover_sitemaps(session: aiohttp.ClientSession, base: str) -> Set[str]:
    urls = set()
    robots_url = urljoin(base, "/robots.txt")
    try:
        async with session.get(robots_url, timeout=ClientTimeout(total=15)) as r:
            if r.status == 200:
                for line in (await r.text()).splitlines():
                    if line.lower().startswith("sitemap:"):
                        urls.add(line.split(":", 1)[1].strip())
    except Exception:
        pass
    urls.add(urljoin(base, "/sitemap.xml"))
    out = set()
    seen = set()
    for sm in urls:
        if sm in seen:
            continue
        seen.add(sm)
        try:
            async with session.get(sm, timeout=ClientTimeout(total=20)) as r:
                if r.status != 200 or "xml" not in (r.headers.get("content-type") or ""):
                    continue
                text = await r.text()
                for loc in re.findall(r"<loc>\s*([^<]+)\s*</loc>", text, re.I):
                    out.add(loc.strip())
        except Exception:
            continue
    return out

# ---------------- Extraction ----------------
def extract_links(base_url: str, html: str) -> Set[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absu = urljoin(base_url, href)
        if not absu.lower().startswith(("http://", "https://")):
            continue
        links.add(canonicalize(absu))
    return links

def extract_images(base_url: str, html: str) -> Set[str]:
    soup = BeautifulSoup(html, "lxml")
    imgs = set()
    for im in soup.find_all("img", src=True):
        src = im["src"].strip()
        absu = urljoin(base_url, src)
        imgs.add(canonicalize(absu))
    md = soup.find("meta", attrs={"property":"og:image"}) or soup.find("meta", attrs={"name":"og:image"})
    if md and md.get("content"):
        imgs.add(canonicalize(urljoin(base_url, md["content"].strip())))
    return {u for u in imgs if u.lower().startswith(("http://","https://"))}

def basic_title_meta(html: str) -> Tuple[str, str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    md = soup.find("meta", attrs={"name":"description"}) or soup.find("meta", attrs={"property":"og:description"})
    meta_desc = md["content"].strip() if md and md.get("content") else ""
    mr = soup.find("meta", attrs={"name":"robots"})
    meta_robots = (mr.get("content") or "").lower() if mr and mr.get("content") else ""
    return title, meta_desc, meta_robots

def pick_canonical(url: str, html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
        link = soup.find("link", rel=lambda v: v and "canonical" in v)
        if link and link.get("href"):
            return canonicalize(urljoin(url, link["href"].strip()))
    except Exception:
        return None
    return None

def extract_jsonld(html: str, base_url: str) -> List[dict]:
    out = []
    try:
        soup = BeautifulSoup(html, "lxml")
        for s in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(s.get_text() or "")
                out.append(data)
            except Exception:
                continue
    except Exception:
        pass
    return out

def extract_record(url: str, status: int, fetched_at: int, content_type: str,
                   title: str, meta_desc: str, meta_robots: str,
                   text: str, html_relpath: str, pdf_relpath: str,
                   images_list: List[dict], out_links: List[str],
                   depth: int, canonical_url: Optional[str],
                   jsonld: List[dict]) -> dict:
    return {
        "url": url,
        "canonical_url": canonical_url or "",
        "title": title,
        "status": status,
        "fetched_at": fetched_at,
        "headers": {"content_type": content_type},
        "meta": {"description": meta_desc, "robots": meta_robots},
        "depth": depth,
        "text": text,
        "html_file": html_relpath,
        "pdf_file": pdf_relpath,
        "images": images_list,
        "out_links": out_links,
        "jsonld": jsonld,
    }

# ---------------- Optional renderer ----------------
class RenderHelper:
    def __init__(self, pattern: Optional[re.Pattern], render_all: bool):
        self.pattern = pattern
        self.render_all = render_all
        self._ready = False
        self._failed = False
        self._playwright = None
        self._browser = None
        self._context = None

    async def ensure(self):
        if (self.pattern or self.render_all) and not self._failed and not self._ready:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=True)
                self._context = await self._browser.new_context()
                self._ready = True
            except Exception:
                self._failed = True

    def should_render(self, url: str) -> bool:
        if not self._ready or self._failed:
            return False
        if self.render_all:
            return True
        return bool(self.pattern and self.pattern.search(url))

    async def fetch_rendered(self, url: str) -> Tuple[int, Optional[str], str]:
        try:
            page = await self._context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=35000)
            html = await page.content()
            await page.close()
            return 200, html, "text/html; rendered"
        except Exception:
            return 0, None, ""

    async def close(self):
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

# ---------------- Crawler ----------------
class SiteDumper:
    def __init__(self, start: str, out_dir: Path, include_subdomains: bool, max_pages: int, concurrency: int,
                 timeout: int, user_agent: str, allow: Optional[re.Pattern], deny: Optional[re.Pattern],
                 delay: float, render_helper: RenderHelper, assets_offsite: bool,
                 gzip_index: bool, ignore_robots: bool, verbose: bool, insecure_ssl: bool,
                 max_depth: Optional[int], stay_on_path: bool, proxy: Optional[str],
                 include_pdfs: bool, respect_meta_robots: bool, state_path: Optional[Path],
                 save_state: bool, export_urls: bool,
                 dataset: bool, dataset_dirname: str):
        self.start = canonicalize(start)
        self.out_dir = out_dir
        self.include_subdomains = include_subdomains
        self.max_pages = max_pages
        self.timeout = timeout
        self.ua = user_agent
        self.allow = allow
        self.deny = deny
        self.delay = delay
        self.render = render_helper
        self.assets_offsite = assets_offsite
        self.gzip_index = gzip_index
        self.ignore_robots = ignore_robots
        self.verbose = verbose
        self.insecure_ssl = insecure_ssl
        self.max_depth = max_depth
        self.stay_on_path = stay_on_path
        self.proxy = proxy
        self.include_pdfs = include_pdfs
        self.respect_meta_robots = respect_meta_robots
        self.state_path = state_path or (out_dir / "state.json")
        self.save_state = save_state
        self.export_urls = export_urls
        self.dataset = dataset
        self.dataset_dir = out_dir / dataset_dirname

        self.sem = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None
        self.rp: Optional[robotparser.RobotFileParser] = None

        p = urlparse(self.start)
        self.domain_root = f"{p.scheme}://{p.netloc}"
        self.start_path = p.path or "/"

        self.queue: "asyncio.Queue[Tuple[str,int]]" = asyncio.Queue()
        self.seen: Set[str] = set()
        self.url_to_id: Dict[str, int] = {}
        self.pages_index: List[dict] = []
        self.graph_edges: Set[Tuple[int,int]] = set()
        self.page_count = 0

        self.assets_dir = self.out_dir / "_assets" / "img"
        self.image_cache: Dict[str, dict] = {}

        self.log_fp = None
        self.urls_fp = None
        self.dataset_pages_fp = None
        self.dataset_text_fp = None
        self.dataset_links_fp = None

        self.retries = 3
        self.backoff_base = 0.5

    async def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        if self.log_fp:
            self.log_fp.write(line)
            self.log_fp.flush()

    def allowed_by_filters(self, url: str) -> bool:
        if SKIP_EXT.match(url):
            if self.include_pdfs and re.search(r"\.pdf(?:$|\?)", url, re.I):
                pass
            else:
                return False
        if self.allow and not self.allow.search(url):
            return False
        if self.deny and self.deny.search(url):
            return False
        if not same_site(self.start, url, self.include_subdomains):
            return False
        if self.stay_on_path:
            pp = urlparse(url).path or "/"
            if not pp.startswith(self.start_path):
                return False
        return True

    async def _get(self, url: str, expect_bytes: bool = False) -> Tuple[int, Optional[bytes], Dict[str,str]]:
        last_exc = None
        headers = {
            "User-Agent": self.ua,
            "Accept": "*/*" if expect_bytes else "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
        for attempt in range(self.retries):
            try:
                async with self.sem:
                    async with self.session.get(
                        url, headers=headers, timeout=ClientTimeout(total=self.timeout),
                        allow_redirects=True, proxy=self.proxy
                    ) as r:
                        hdrs = {k.lower(): v for k, v in r.headers.items()}
                        if expect_bytes:
                            body = await r.read()
                        else:
                            body = (await r.text(errors="ignore")).encode("utf-8", "ignore")
                        return r.status, body, hdrs
            except (asyncio.TimeoutError, ClientConnectorError) as e:
                last_exc = e
            except Exception as e:
                last_exc = e
            await asyncio.sleep(self.backoff_base * (2 ** attempt))
        await self.log(f"RETRY FAIL {url} {repr(last_exc)}") if self.verbose else None
        return 0, None, {}

    async def fetch_http(self, url: str) -> Tuple[int, Optional[str], str]:
        status, body, hdrs = await self._get(url, expect_bytes=False)
        if body is None:
            return status, None, ""
        ct = (hdrs.get("content-type") or "").lower()
        text = body.decode("utf-8", "ignore")
        is_html = ("text/html" in ct) or ("text/" in ct and "<html" in text[:4096].lower())
        if is_html:
            return status, text, ct or "text/html"
        if self.include_pdfs and ("application/pdf" in ct or re.search(r"\.pdf(?:$|\?)", url, re.I)):
            return status, None, ct or "application/pdf"
        return status, None, ct

    async def fetch(self, url: str) -> Tuple[int, Optional[str], str]:
        status, html, ct = await self.fetch_http(url)
        if (not html or status != 200) and self.render.should_render(url):
            rstatus, rhtml, rct = await self.render.fetch_rendered(url)
            if rhtml:
                return rstatus, rhtml, rct
        return status, html, ct

    async def download_pdf(self, url: str, save_path: Path) -> Tuple[int, Optional[str]]:
        status, body, hdrs = await self._get(url, expect_bytes=True)
        if status == 200 and body:
            ensure_dir(save_path.parent)
            with open(save_path, "wb") as f:
                f.write(body)
            text = ""
            if pdf_extract_text:
                try:
                    text = pdf_extract_text(str(save_path)) or ""
                except Exception:
                    text = ""
            return 200, text
        return status, None

    async def download_image(self, img_url: str) -> Optional[dict]:
        if img_url in self.image_cache:
            return self.image_cache[img_url]
        if not self.assets_offsite and not same_site(self.start, img_url, self.include_subdomains):
            return None
        status, body, hdrs = await self._get(img_url, expect_bytes=True)
        ct = (hdrs.get("content-type") or "").lower()
        if status != 200 or not ct.startswith("image/") or not body:
            meta = {"original_url": img_url, "status": status, "content_type": ct}
            self.image_cache[img_url] = meta
            return meta
        sha = hashlib.sha256(body).hexdigest()
        ext = guess_ext(img_url, ct, default="img")
        relpath = Path("_assets") / "img" / f"{sha}.{ext}"
        abspath = self.out_dir / relpath
        if not abspath.exists():
            ensure_dir(abspath.parent)
            with open(abspath, "wb") as f:
                f.write(body)
        meta = {"original_url": img_url, "saved_path": str(relpath).replace("\\","/"),
                "sha256": sha, "content_type": ct, "status": 200}
        self.image_cache[img_url] = meta
        return meta

    async def process_page(self, url: str, depth: int):
        try:
            if len(self.seen) >= self.max_pages:
                return
            if url in self.seen:
                return
            self.seen.add(url)

            is_seed = (url == self.start)
            if not is_seed and not self.allowed_by_filters(url):
                if self.verbose: await self.log(f"SKIP FILTER {url}")
                return
            if self.max_depth is not None and depth > self.max_depth:
                if self.verbose: await self.log(f"SKIP DEPTH>{self.max_depth} {url}")
                return
            if not self.ignore_robots and not self.rp.can_fetch(self.ua, url):
                if self.verbose: await self.log(f"BLOCK ROBOTS {url}")
                if not is_seed:
                    return

            if self.delay > 0:
                await asyncio.sleep(self.delay)

            status, html, ct = await self.fetch(url)
            if self.verbose: await self.log(f"FETCH {status} {ct or ''} {url}")

            ts = int(time.time())
            self.page_count += 1
            pid = self.page_count
            self.url_to_id[url] = pid

            slug = slugify_from_url(url)
            page_dir = self.out_dir / f"{pid:05d}-{slug}"
            ensure_dir(page_dir)

            out_links: List[str] = []
            images_list: List[dict] = []
            title = ""
            meta_desc = ""
            meta_robots = ""
            text = ""
            html_rel = ""
            pdf_rel = ""
            canonical_url = None
            jsonld = []

            if self.include_pdfs and (ct and "application/pdf" in ct or re.search(r"\.pdf(?:$|\?)", url, re.I)):
                pdf_path = page_dir / "page.pdf"
                pstatus, ptext = await self.download_pdf(url, pdf_path)
                if pstatus == 200:
                    pdf_rel = os.path.relpath(pdf_path, self.out_dir).replace("\\","/")
                    text = clean_text(ptext or "")
            elif html:
                canonical_url = pick_canonical(url, html)
                html_path = page_dir / "page.html"
                with open(html_path, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(html)
                html_rel = os.path.relpath(html_path, self.out_dir).replace("\\","/")

                title, meta_desc, meta_robots = basic_title_meta(html)
                # primary extraction
                t_primary = trafilatura.extract(html, url=url, include_comments=False, include_tables=True) or ""
                # fallback if empty
                if not t_primary.strip():
                    soup = BeautifulSoup(html, "lxml")
                    t_primary = soup.get_text(separator="\n", strip=True)
                text = clean_text(t_primary)
                jsonld = extract_jsonld(html, url)

                follow_links = True
                if self.respect_meta_robots and "nofollow" in meta_robots:
                    follow_links = False

                if follow_links:
                    for link in extract_links(url, html):
                        if not self.allowed_by_filters(link):
                            continue
                        if same_site(self.start, link, self.include_subdomains):
                            out_links.append(link)
                            if link not in self.seen:
                                await self.queue.put((link, depth + 1))

                for img_url in extract_images(url, html):
                    meta = await self.download_image(img_url)
                    if meta:
                        images_list.append(meta)

            rec = extract_record(
                url=url, status=status, fetched_at=ts, content_type=ct or "",
                title=title, meta_desc=meta_desc, meta_robots=meta_robots,
                text=text, html_relpath=html_rel, pdf_relpath=pdf_rel,
                images_list=images_list, out_links=out_links, depth=depth,
                canonical_url=canonical_url, jsonld=jsonld,
            )
            with open(page_dir / "page.json", "w", encoding="utf-8") as jf:
                json.dump(rec, jf, ensure_ascii=False, indent=2)

            # stream dataset rows
            if self.dataset and self.dataset_pages_fp and self.dataset_text_fp and self.dataset_links_fp:
                row = {
                    "id": pid,
                    "url": url,
                    "canonical_url": canonical_url or "",
                    "title": title,
                    "text": text,
                    "meta_description": meta_desc,
                    "meta_robots": meta_robots,
                    "status": status,
                    "fetched_at": ts,
                    "depth": depth,
                    "html_file": html_rel,
                    "pdf_file": pdf_rel,
                    "image_paths": [im.get("saved_path","") for im in images_list if isinstance(im, dict) and im.get("saved_path")],
                    "out_links": out_links,
                    "jsonld": jsonld,
                }
                self.dataset_pages_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.dataset_text_fp.write(json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False) + "\n")
                for to_url in out_links:
                    self.dataset_links_fp.write(json.dumps({"from": url, "to": to_url}, ensure_ascii=False) + "\n")

            if self.export_urls and self.urls_fp:
                self.urls_fp.write(url + "\n")

            self.pages_index.append({
                "id": pid, "url": url, "title": title, "status": status,
                "folder": page_dir.name, "fetched_at": ts, "depth": depth
            })
        except Exception as e:
            if self.verbose:
                await self.log("EXC " + repr(e))
                await self.log(traceback.format_exc())

    async def worker(self):
        while True:
            item = await self.queue.get()
            try:
                url, depth = item
                await self.process_page(url, depth)
            finally:
                self.queue.task_done()

    def _load_state(self):
        try:
            if self.state_path and self.state_path.exists():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.seen = set(data.get("seen", []))
                pending = data.get("pending", [])
                for u, d in pending:
                    self.queue.put_nowait((u, d))
                self.page_count = data.get("page_count", 0)
        except Exception:
            pass

    def _save_state(self):
        if not self.save_state:
            return
        try:
            pending = list(self.queue._queue)  # snapshot
            data = {"seen": sorted(self.seen), "pending": pending, "page_count": self.page_count}
            ensure_dir(self.state_path.parent)
            self.state_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass

    async def run(self):
        ensure_dir(self.out_dir)
        self.log_fp = open(self.out_dir / "crawl.log", "a", encoding="utf-8")
        if self.export_urls:
            self.urls_fp = open(self.out_dir / "urls.txt", "w", encoding="utf-8")
        if self.dataset:
            ensure_dir(self.dataset_dir)
            self.dataset_pages_fp = open(self.dataset_dir / "pages.jsonl", "w", encoding="utf-8")
            self.dataset_text_fp  = open(self.dataset_dir / "text.jsonl", "w", encoding="utf-8")
            self.dataset_links_fp = open(self.dataset_dir / "links.jsonl", "w", encoding="utf-8")
        await self.log(f"START {self.start}")

        connector = aiohttp.TCPConnector(limit=0) if not self.insecure_ssl else aiohttp.TCPConnector(limit=0, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            self.session = session
            self.rp = await load_robots(session, self.domain_root)
            await self.render.ensure()

            self._load_state()
            if self.queue.qsize() == 0:
                try:
                    for u in await discover_sitemaps(session, self.domain_root):
                        cu = canonicalize(u)
                        if self.allowed_by_filters(cu):
                            await self.queue.put((cu, 0))
                            if self.verbose: await self.log(f"SEED SITEMAP {cu}")
                except Exception:
                    pass
                await self.queue.put((self.start, 0))
                if self.verbose: await self.log(f"SEED START {self.start}")

            worker_count = max(4, min(32, self.sem._value))
            tasks = [asyncio.create_task(self.worker()) for _ in range(worker_count)]
            try:
                await self.queue.join()
            finally:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._save_state()

        await self.render.close()

        url_to_pid = self.url_to_id
        edges: Set[Tuple[int,int]] = set()
        for page in self.pages_index:
            folder = page["folder"]
            try:
                with open(self.out_dir / folder / "page.json", "r", encoding="utf-8") as jf:
                    rec = json.load(jf)
                for ol in rec.get("out_links", []):
                    if ol in url_to_pid:
                        edges.add((page["id"], url_to_pid[ol]))
            except Exception:
                continue

        self.graph_edges = edges
        graph = {
            "nodes": [{"id": p["id"], "url": p["url"], "title": p.get("title",""), "depth": p.get("depth", 0)} for p in self.pages_index],
            "edges": [{"from": a, "to": b} for (a,b) in sorted(self.graph_edges)],
        }
        with open(self.out_dir / "graph.json", "w", encoding="utf-8") as gf:
            json.dump(graph, gf, ensure_ascii=False, indent=2)

        idx_path = self.out_dir / ("index.jsonl.gz" if self.gzip_index else "index.jsonl")
        if self.gzip_index:
            with gzip.open(idx_path, "wb") as zf:
                for row in self.pages_index:
                    zf.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        else:
            with open(idx_path, "w", encoding="utf-8") as f:
                for row in self.pages_index:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        await self.log(f"DONE pages={len(self.pages_index)} assets={len(self.image_cache)}")
        for fp in (self.log_fp, self.urls_fp, self.dataset_pages_fp, self.dataset_text_fp, self.dataset_links_fp):
            try:
                fp and fp.close()
            except Exception:
                pass

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Whole-site dumper with per-page folders and JSONL dataset.")
    ap.add_argument("start_url", help="e.g., https://example.com")
    ap.add_argument("--out-dir", default="site_dump", help="Output directory")
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--include-subdomains", action="store_true")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--user-agent", default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36")
    ap.add_argument("--allow", help="Regex. Only crawl URLs matching this.")
    ap.add_argument("--deny", help="Regex. Skip URLs matching this.")
    ap.add_argument("--delay-ms", type=int, default=0, help="Politeness delay per request.")
    ap.add_argument("--render-pattern", help="Regex. Render matching URLs with Playwright.")
    ap.add_argument("--render-all", action="store_true", help="Render every page with Playwright.")
    ap.add_argument("--assets-offsite", action="store_true", help="Also download offsite images.")
    ap.add_argument("--gzip-index", action="store_true", help="Write index.jsonl.gz")
    ap.add_argument("--ignore-robots", action="store_true", help="Do not enforce robots.txt (use only if permitted).")
    ap.add_argument("--verbose", action="store_true", help="Write detailed crawl.log entries.")
    ap.add_argument("--insecure", action="store_true", help="Disable SSL verification.")
    ap.add_argument("--max-depth", type=int, help="Limit crawl depth from the start URL.")
    ap.add_argument("--stay-on-path", action="store_true", help="Stay under the start URL path.")
    ap.add_argument("--proxy", help="HTTP/HTTPS proxy URL for requests.")
    ap.add_argument("--include-pdfs", action="store_true", help="Fetch PDFs and extract text when possible.")
    ap.add_argument("--respect-meta-robots", action="store_true", help="Obey <meta name='robots' content='nofollow'> for links.")
    ap.add_argument("--resume", help="Resume file path (defaults to out_dir/state.json).")
    ap.add_argument("--no-save-state", action="store_true", help="Do not write state.json at the end.")
    ap.add_argument("--export-urls", action="store_true", help="Write URLs to urls.txt")
    # NEW: dataset export
    ap.add_argument("--dataset", action="store_true", help="Also write JSONL dataset under _dataset/")
    ap.add_argument("--dataset-dir", default="_dataset", help="Subfolder name for dataset JSONL files")
    return ap.parse_args()

def main():
    args = parse_args()
    allow = re.compile(args.allow) if args.allow else None
    deny  = re.compile(args.deny)  if args.deny  else None
    render_pat = re.compile(args.render_pattern) if args.render_pattern else None
    delay = max(0.0, args.delay_ms / 1000.0)
    out_dir = Path(args.out_dir).resolve()
    state_path = Path(args.resume).resolve() if args.resume else (out_dir / "state.json")

    dumper = SiteDumper(
        start=canonicalize(args.start_url),
        out_dir=out_dir,
        include_subdomains=args.include_subdomains,
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        timeout=args.timeout,
        user_agent=args.user_agent,
        allow=allow,
        deny=deny,
        delay=delay,
        render_helper=RenderHelper(render_pat, args.render_all),
        assets_offsite=args.assets_offsite,
        gzip_index=args.gzip_index,
        ignore_robots=args.ignore_robots,
        verbose=args.verbose,
        insecure_ssl=args.insecure,
        max_depth=args.max_depth,
        stay_on_path=args.stay_on_path,
        proxy=args.proxy,
        include_pdfs=args.include_pdfs,
        respect_meta_robots=args.respect_meta_robots,
        state_path=state_path,
        save_state=not args.no_save_state,
        export_urls=args.export_urls,
        dataset=args.dataset,
        dataset_dirname=args.dataset_dir,
    )
    try:
        asyncio.run(dumper.run())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)

if __name__ == "__main__":
    main()
