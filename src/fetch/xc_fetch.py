#!/usr/bin/env python3
import argparse, re, time, pathlib, requests, os, sys, random

# --- XC endpoints -------------------------------------------------------------
V3 = "https://xeno-canto.org/api/3/recordings"
V2 = "https://xeno-canto.org/api/2/recordings"

# behavior toggles (via env)
ALLOW_ND = os.environ.get("XC_ALLOW_ND", "0") == "1"   # include No-Derivatives if 1
PER_PAGE = 300
DEF_QUERY = ""  # append extra tag filters via --extra_query

# --- helpers ------------------------------------------------------------------
def slug(s: str) -> str:
    return re.sub(r"[^-_.A-Za-z0-9]+", "_", s).strip("_")

def _sleep_with_jitter(base: float) -> None:
    time.sleep(base + random.uniform(0, base / 2.0))

# --- http / retry -------------------------------------------------------------
def get_json(session: requests.Session, url: str, params: dict, max_retries: int = 6):
    delay = 0.6
    for attempt in range(1, max_retries + 1):
        r = session.get(url, params=params, timeout=60)
        if r.status_code in (429, 503) or 500 <= r.status_code < 600:
            ra = r.headers.get("Retry-After", "").strip()
            try:
                wait = float(ra)
            except Exception:
                wait = delay
            print(f"   .. {r.status_code} from server, backing off {wait:.1f}s (try {attempt}/{max_retries})")
            _sleep_with_jitter(wait)
            delay = min(delay * 1.8, 10.0)
            continue
        if r.status_code != 200:
            body = r.text[:200].replace("\n", " ")
            raise RuntimeError(f"HTTP {r.status_code}: {body}")
        try:
            j = r.json()
        except Exception:
            raise RuntimeError(f"Non-JSON response: {r.text[:200]}")
        if isinstance(j, dict) and ("error" in j or "warning" in j):
            print(f"   !! server says: {j}")
        return j
    raise RuntimeError(f"Gave up after {max_retries} retries on {url}")

def page_mp3_urls(session: requests.Session, rec_id: str, ua: str):
    """
    Fetch https://xeno-canto.org/<id> and extract candidate download URLs.
    Returns a (possibly empty) list of absolute https URLs.
    """
    if not rec_id:
        return []
    base = f"https://xeno-canto.org/{rec_id}"
    try:
        r = session.get(base, timeout=30, headers={
            "User-Agent": ua,
            "Accept": "text/html,*/*;q=0.8",
            "Referer": "https://xeno-canto.org/",
        })
        r.raise_for_status()
        html = r.text

        urls = set()

        # canonical download link(s)
        for m in re.finditer(r'href="(/'+re.escape(rec_id)+r'/download[^"]*)"', html):
            urls.add("https://xeno-canto.org" + m.group(1))

        # direct MP3 link in href
        for m in re.finditer(r'href="(//(?:www\.)?xeno-canto\.org/[^"]+\.mp3[^"]*)"', html):
            u = m.group(1)
            if u.startswith("//"): u = "https:" + u
            urls.add(u)

        # data attributes carrying file link
        for m in re.finditer(r'data-(?:file|xc-file)="([^"]+\.mp3[^"]*)"', html):
            u = m.group(1)
            if u.startswith("//"): u = "https:" + u
            elif u.startswith("/"): u = "https://xeno-canto.org" + u
            urls.add(u)

        # add www. variants
        urls |= {u.replace("://xeno-canto.org/", "://www.xeno-canto.org/") for u in urls}
        return list(urls)
    except Exception:
        return []

def download_with_retry(session: requests.Session, url: str, dest_path: pathlib.Path, rec_id: str = "", max_retries: int = 6) -> bool:
    """
    Robust XC audio downloader:
      - Browser-like UA + Referer
      - Prime cookies via /<id>
      - Try multiple URL variants (original, www., /<id>/download, ?type=mp3)
      - If needed, parse /<id> HTML to find download link(s)
      - Backoff on 429/5xx; log distinct failure reasons; drop tiny stubs
    """
    delay = 0.9
    ua = os.environ.get(
        "XC_FETCH_UA",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36",
    )
    hdrs = {
        "Accept": "audio/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": ua,
        "Referer": "https://xeno-canto.org/",
        "Connection": "keep-alive",
    }

    def normalize(u: str) -> str:
        if not u:
            return u
        if u.startswith("//"):
            u = "https:" + u
        u = u.replace("http://", "https://")
        return u

    url = normalize(url)

    def prime_cookies():
        if not rec_id:
            return
        try:
            session.get(
                f"https://xeno-canto.org/{rec_id}",
                timeout=30,
                headers={"User-Agent": ua, "Accept": "text/html,*/*;q=0.8", "Referer": "https://xeno-canto.org/"},
            )
        except Exception:
            pass

    def candidate_urls():
        C = []
        if url:
            C.append(url)
            C.append(url.replace("://xeno-canto.org/", "://www.xeno-canto.org/"))
        if rec_id:
            bases = [f"https://xeno-canto.org/{rec_id}", f"https://www.xeno-canto.org/{rec_id}"]
            for b in bases:
                C.append(f"{b}/download")
                C.append(f"{b}/download?type=mp3")
        # dedupe preserving order
        seen, out = set(), []
        for u in C:
            if u and u not in seen:
                out.append(u); seen.add(u)
        return out

    tried_any = False
    for attempt in range(1, max_retries + 1):
        if attempt == 1 and rec_id:
            prime_cookies()

        for candidate in candidate_urls():
            tried_any = True
            try:
                with session.get(candidate, stream=True, timeout=300, allow_redirects=True, headers=hdrs) as r:
                    status = r.status_code

                    if status in (429, 503) or 500 <= status < 600:
                        ra = r.headers.get("Retry-After", "").strip()
                        try:
                            wait = float(ra) if ra else delay
                        except Exception:
                            wait = delay
                        print(f"         {status} on {candidate}, retry {wait:.1f}s (try {attempt}/{max_retries})")
                        time.sleep(wait + random.uniform(0, wait / 2))
                        delay = min(delay * 1.8, 15.0)
                        break  # try again with backoff

                    if status in (401, 403):
                        print(f"         {status} for {candidate} — will try alternates")
                        # try next candidate
                        continue

                    if status == 404:
                        print(f"         404 for {candidate}")
                        continue

                    r.raise_for_status()

                    ct = (r.headers.get("Content-Type") or "").lower()
                    if not any(x in ct for x in ("audio", "mpeg", "mp3", "ogg", "octet-stream")):
                        # non-audio; try another candidate
                        dbg = dest_path.with_suffix(".err.html")
                        try:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(dbg, "wb") as df:
                                df.write(r.content[:4000])
                        except Exception:
                            pass
                        print(f"         non-audio ({ct or 'no Content-Type'}) from {candidate} — saved {dbg.name}")
                        continue

                    # stream to disk
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    size = 0
                    with open(dest_path, "wb") as f:
                        for chunk in r.iter_content(1 << 14):
                            if chunk:
                                f.write(chunk); size += len(chunk)

                    if size < 8192:
                        print(f"         very small file ({size} bytes) from {candidate} — dropping")
                        try: os.remove(dest_path)
                        except Exception: pass
                        continue

                    return True  # success

            except requests.exceptions.RequestException as e:
                print(f"         network error on {candidate}: {e}")
                continue

        # HTML scrape fallback once per attempt
        if rec_id:
            scraped = page_mp3_urls(session, rec_id, ua)
            if scraped:
                print(f"         trying {len(scraped)} URL(s) scraped from /{rec_id}")
                for s in scraped:
                    try:
                        with session.get(s, stream=True, timeout=300, allow_redirects=True, headers=hdrs) as r:
                            if r.status_code == 200:
                                ct = (r.headers.get("Content-Type") or "").lower()
                                if any(x in ct for x in ("audio", "mpeg", "mp3", "ogg", "octet-stream")):
                                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                                    size = 0
                                    with open(dest_path, "wb") as f:
                                        for chunk in r.iter_content(1 << 14):
                                            if chunk:
                                                f.write(chunk); size += len(chunk)
                                    if size >= 8192:
                                        return True
                                    else:
                                        try: os.remove(dest_path)
                                        except Exception: pass
                            else:
                                print(f"         scraped URL {s} -> HTTP {r.status_code}")
                    except Exception as e:
                        print(f"         error on scraped URL {s}: {e}")

        time.sleep(delay + random.uniform(0, delay / 2))
        delay = min(delay * 1.8, 15.0)

    if not tried_any:
        print(f"         no candidate URLs for {rec_id or url}")
    print(f"         giving up on {rec_id or url}")
    return False

# --- species list -------------------------------------------------------------
def parse_species_list(path: str):
    """
    Expect BirdNET-style lines: 'Genus species[_subspecies]_Common Name'
    Returns tuples: (genus, species_or_rest, common, full_label)
    """
    labels, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "_" not in s:
                continue
            if s in seen:
                continue
            seen.add(s)
            sci, com = s.split("_", 1)
            parts = sci.split()
            if len(parts) < 2:
                continue
            genus, species = parts[0], " ".join(parts[1:])
            labels.append((genus, species, com, f"{genus} {species}_{com}"))
    return labels

# --- count + fetch ------------------------------------------------------------
def count_any(session: requests.Session, key: str, q: str):
    """Try v3 first; fall back to v2 if needed."""
    try:
        j = get_json(session, V3, {"query": q, "key": key, "per_page": PER_PAGE, "page": 1})
        n = int(j.get("numRecordings", 0) or 0)
        pages = int(j.get("numPages", 0) or 0)
        return ("v3", q, n, pages)
    except Exception as e:
        print(f"   !! v3 error: {e}")
    try:
        j = get_json(session, V2, {"query": q, "page": 1})
        n = int(j.get("numRecordings", 0) or 0)
        pages = int(j.get("numPages", 0) or 0)
        return ("v2", q, n, pages)
    except Exception as e:
        print(f"   !! v2 error: {e}")
        return ("none", q, 0, 0)

def fetch_pages(session: requests.Session, api: str, key: str, q: str, out_dir: pathlib.Path, pages: int) -> int:
    downloaded = 0
    skipped_existing = 0
    skipped_nd = 0
    skipped_missing = 0

    for page in range(1, pages + 1):
        params = {"query": q, "per_page": PER_PAGE, "page": page}
        if api == "v3":
            params["key"] = key

        j = get_json(session, V3 if api == "v3" else V2, params)

        for rec in (j.get("recordings", []) or []):
            lic = (rec.get("lic") or rec.get("license") or "").lower()
            if ("/nd/" in lic or "no derivatives" in lic) and not ALLOW_ND:
                skipped_nd += 1
                continue

            file_url = rec.get("file") or ""
            if file_url.startswith("//"):
                file_url = "https:" + file_url
            if not file_url:
                skipped_missing += 1
                continue

            rid = str(rec.get("id") or slug((rec.get("gen", "") + rec.get("sp", "") + rec.get("rec", ""))))
            mp3 = out_dir / f"xc{rid}.mp3"

            if mp3.exists():
                try:
                    if mp3.stat().st_size < 8192:
                        print(f"      existing but tiny -> re-download: {mp3}")
                        mp3.unlink(missing_ok=True)
                    else:
                        skipped_existing += 1
                        continue
                except Exception:
                    pass

            ok = download_with_retry(session, file_url, mp3, rid)
            if ok:
                downloaded += 1
                if downloaded % 5 == 0:
                    print(f"      ...{downloaded} files")
            else:
                skipped_missing += 1

        _sleep_with_jitter(0.3)

    print(f"      summary: +{downloaded} new, {skipped_existing} existing, {skipped_nd} ND, {skipped_missing} missing/forbidden")
    return downloaded

# --- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species_list", required=True, help="Path to BirdNET-style list: 'Genus species_Common Name'")
    ap.add_argument("--out_raw", default="data_raw", help="Output root for raw MP3s")
    ap.add_argument("--extra_query", default=DEF_QUERY, help="Extra XC tags, e.g. ' type:song len:5-120'")
    args = ap.parse_args()

    key = os.environ.get("XC_API_KEY", "").strip()
    if not key:
        print("ERROR: XC_API_KEY not set. Run: export XC_API_KEY=YOUR_KEY", file=sys.stderr)
        sys.exit(2)

    session = requests.Session()
    session.headers.update({
        "User-Agent": os.environ.get(
            "XC_FETCH_UA",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36"
        )
    })

    total = 0
    for genus, species, common, full_label in parse_species_list(args.species_list):
        out_dir = pathlib.Path(args.out_raw) / slug(full_label)
        out_dir.mkdir(parents=True, exist_ok=True)

        base = f'gen:{genus} sp:{species}'
        q_candidates = [
            f'{base}{args.extra_query}',
            f'sp:"{genus} {species}"{args.extra_query}',
            f'en:"{common}"{args.extra_query}' if common else None,
        ]
        q_candidates = [q for q in q_candidates if q]

        print(f"[XC] {full_label}")
        picked_api, picked_q, n, pages = ("none", "", 0, 0)
        for qi in q_candidates:
            api, q_used, n_i, pages_i = count_any(session, key, qi)
            print(f"   -> {api} {qi!r}: {n_i} recs, {pages_i} pages")
            if api != "none" and n_i > 0:
                picked_api, picked_q, n, pages = api, q_used, n_i, pages_i
                break

        if n == 0:
            print("   -> no results; skipping")
            continue

        got = fetch_pages(session, picked_api, key, picked_q, out_dir, pages)
        print(f"   -> downloaded {got} files")
        total += got

    print(f"\nDone. Total new files: {total}")

if __name__ == "__main__":
    main()
