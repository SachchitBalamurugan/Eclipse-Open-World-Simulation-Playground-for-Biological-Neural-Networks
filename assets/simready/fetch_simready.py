"""Fetch complete NVIDIA SimReady asset folders from HF (nvidia/simready-assets) with retries/resume."""
import sys, os, time, json, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import requests

REPO = "nvidia/simready-assets"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/"
RES = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
S = requests.Session()

def get(url, **kw):
    for i in range(8):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {r.status_code}")
            return r
        except requests.RequestException as e:
            w = min(60, 2 ** i); print(f"  retry {i} {url[-60:]}: {e}; sleep {w}", flush=True); time.sleep(w)
    raise RuntimeError("giving up " + url)

def listing(path, recursive=True):
    out, url = [], API + urllib.parse.quote(path) + ("?recursive=1" if recursive else "")
    while url:
        r = get(url); r.raise_for_status(); out += r.json()
        nxt = r.links.get("next", {}).get("url"); url = nxt
    return out

def download(path, size, prefix_strip):
    dst = os.path.join(OUT, path[len(prefix_strip):].lstrip("/")) if prefix_strip else os.path.join(OUT, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(10):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have == size: return
        if have > size: os.remove(dst); have = 0
        try:
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with S.get(RES + urllib.parse.quote(path), headers=hdr, stream=True, timeout=60) as r:
                if r.status_code not in (200, 206): raise requests.RequestException(f"HTTP {r.status_code}")
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(dst, mode) as f:
                    for ch in r.iter_content(1 << 16): f.write(ch)
        except requests.RequestException as e:
            w = min(60, 2 ** i); print(f"  dl retry {i} {path}: {e}", flush=True); time.sleep(w)
    if os.path.getsize(dst) != size: raise RuntimeError("incomplete " + path)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ls":
        for e in listing(sys.argv[2] if len(sys.argv) > 2 else "", recursive=False):
            print(e["type"], e["path"], e.get("size"))
    elif cmd == "size":
        for p in sys.argv[2:]:
            fs = [e for e in listing(p) if e["type"] == "file"]
            print(p, len(fs), "files", round(sum(e["size"] for e in fs) / 1e6, 1), "MB", flush=True)
    elif cmd == "get":
        # Textures (.png) are skipped by default (set SIMREADY_TEXTURES=1 to include): the MuJoCo/GLB
        # conversion uses flat material colours, and PNGs are ~99% of each folder (wheelbarrow: 558 MB PNG vs 3.5 MB USD).
        skip_png = os.environ.get("SIMREADY_TEXTURES") != "1"
        for p in sys.argv[2:]:
            fs = [e for e in listing(p) if e["type"] == "file" and not (skip_png and e["path"].endswith(".png"))]
            parent = p.rsplit("/", 1)[0] if "/" in p else ""
            print(p, len(fs), "files", flush=True)
            with ThreadPoolExecutor(2) as ex:
                list(ex.map(lambda e: download(e["path"], e["size"], parent), fs))
            print("done", p, flush=True)
