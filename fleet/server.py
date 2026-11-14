"""Fleet dashboard server for Freestyle VMs running NeuroRover bench shards. Stdlib only.
   .venv/Scripts/python fleet/server.py   ->  http://127.0.0.1:8770
"""
import json, os, shutil, subprocess, tarfile, threading, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = "resources/challenges/freeze-preparation/09-freeze/data.json.gz"
STATE = HERE / "state.json"
NPX = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
FS = [NPX, "-y", "freestyle@latest"]
PRICE = 0.27
EVENTS = deque(maxlen=400)
LOCK = threading.Lock()
CACHE = {"vms": [], "runningCount": 0, "ts": 0, "error": None}
PROBES = {}          # vm id -> last good probe
BUSY = {"launch": False}
POOL = ThreadPoolExecutor(max_workers=12)
PROBE_POOL = ThreadPoolExecutor(max_workers=10)


def log(msg, level="info"):
    EVENTS.append({"t": time.time(), "level": level, "msg": msg})
    print(time.strftime("%H:%M:%S"), level, msg, flush=True)


def fs(*args, timeout=330, retries=2):
    """Run the freestyle CLI; returns (rc, stdout, stderr). Retries on failure."""
    last = (1, "", "")
    for attempt in range(retries + 1):
        try:
            p = subprocess.run(FS + list(args), cwd=str(ROOT), capture_output=True, text=True,
                               timeout=timeout, encoding="utf-8", errors="replace",
                               stdin=subprocess.DEVNULL)
            last = (p.returncode, p.stdout or "", p.stderr or "")
            if p.returncode == 0:
                return last
            if "LIMIT_EXCEEDED" in (last[1] + last[2]):
                return last
        except subprocess.TimeoutExpired:
            last = (124, "", "timeout")
        except Exception as e:  # noqa
            last = (1, "", repr(e))
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return last


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def parse_json(out):
    i = out.find("{")
    return json.loads(out[i:]) if i >= 0 else {}


def parse_probe(out):
    d = {"ok": True, "log": []}
    head, _, tail = out.partition("---LOG---")
    for line in head.splitlines():
        k, _, v = line.strip().partition("=")
        try:
            if k == "R": d["results"] = int(v)
            elif k == "P": d["procs"] = int(v)
            elif k == "N": d["nproc"] = int(v)
            elif k == "L": d["load"] = [float(x) for x in v.split()[:3]]
            elif k == "M":
                a, b = v.split(); d["mem_used_mb"], d["mem_total_mb"] = int(a), int(b)
        except Exception:
            pass
    d["log"] = [l.rstrip()[-240:] for l in tail.replace("\r", "\n").splitlines() if l.strip()][-5:]
    if "nproc" not in d:
        d["ok"] = False
    return d


def probe(vm):
    vid = vm["id"]
    rc, out, err = fs("vm", "exec", vid, "--", "bash", "/home/ubuntu/probe.sh", timeout=60, retries=0)
    d = parse_probe(out)
    if not d.get("ok") and "No such file" in (out + err):
        # probe script missing (VM not created by us, or still uploading): upload and retry once
        fs("vm", "scp", "fleet/probe.sh", f"{vid}:/home/ubuntu/probe.sh", timeout=60, retries=0)
        rc, out, err = fs("vm", "exec", vid, "--", "bash", "/home/ubuntu/probe.sh", timeout=60, retries=0)
        d = parse_probe(out)
    if not d.get("ok"):
        d["error"] = (err or out).strip()[-200:]
    d["ts"] = time.time()
    return vid, d


def refresh():
    rc, out, err = fs("vm", "list", "--output", "json", timeout=60, retries=1)
    if rc != 0:
        CACHE["error"] = (err or out)[-300:]
        return
    try:
        data = parse_json(out)
    except Exception as e:
        CACHE["error"] = "bad json: %r" % e
        return
    vms = data.get("vms", [])
    st = load_state()
    shards = st.get("shards", {})
    targets = [v for v in vms if str(v.get("slug") or "").startswith("nr-")
               and str(v.get("state", "running")).lower() == "running"]
    for vid, d in PROBE_POOL.map(probe, targets):
        if d.get("ok") or vid not in PROBES or not PROBES[vid].get("ok"):
            PROBES[vid] = d
        else:
            PROBES[vid] = dict(PROBES[vid], stale=True)
    for v in vms:
        v["probe"] = PROBES.get(v["id"])
        sh = shards.get(v["id"])
        if sh:
            v["shard"], v["nshards"], v["expected"], v["phase"] = sh["shard"], st.get("n_vms"), sh.get("expected"), sh.get("phase")
    launch = {k: st.get(k) for k in ("launched_at", "condition", "seeds", "n_vms", "tag")} if st else None
    with LOCK:
        CACHE.update(data)
        CACHE.update(vms=vms, ts=time.time(), error=None, launch=launch, price_per_vm_hour=PRICE)


def refresher():
    while True:
        t0 = time.time()
        try:
            refresh()
        except Exception as e:
            log("refresh error: %r" % e, "error")
        time.sleep(max(1.0, 8 - (time.time() - t0)))


def build_payload():
    run = lambda *a: subprocess.run(list(a), cwd=str(ROOT), capture_output=True, text=True).stdout.splitlines()
    files = run("git", "ls-files")
    files += [f for f in run("git", "ls-files", "--others", "--exclude-standard", "neurorover") if f.endswith(".py")]
    files = [f for f in files if not f.startswith(("neurorover/results/", "fleet/", "runs/"))]
    files.append(DATA)
    out = HERE / "payload.tar.gz"
    n = 0
    with tarfile.open(out, "w:gz", compresslevel=4) as tf:
        for f in dict.fromkeys(files):
            p = ROOT / f
            if p.is_file():
                tf.add(str(p), arcname=f); n += 1
    log("payload built from working tree: %d files, %.1f MB" % (n, out.stat().st_size / 1e6))


def launch_one(i, n, condition, seeds, tag, st, reuse=None):
    slug = "nr-fleet-%d" % i
    if reuse is not None:
        rc, out, err = (0, json.dumps({"id": reuse[slug]}), "") if slug in reuse else (1, "", "no existing VM to reuse")
    else:
      rc, out, err = fs("vm", "create", "--slug", slug, "--replace-slug", "--no-ssh", "--idle-timeout-seconds", "7200",
                      "--metadata", "project=neurorover", "--output", "json", timeout=180, retries=2)
    if rc != 0:
        log("%s create failed: %s" % (slug, (err or out).strip()[-200:]), "error"); return
    try:
        d = parse_json(out)
        vid = d.get("id") or d.get("vmId") or (d.get("vm") or {}).get("id")
    except Exception:
        vid = None
    if not vid:
        log("%s: no id in create output: %s" % (slug, out[-200:]), "error"); return
    expected = len(range(i, seeds, n))
    with LOCK:
        st["shards"][vid] = {"shard": i, "slug": slug, "expected": expected, "phase": "uploading"}
        save_state(st)
    log("%s created (%s), uploading payload" % (slug, vid))
    for src, dst in (("fleet/probe.sh", "probe.sh"), ("fleet/start.sh", "start.sh"),
                     ("fleet/collect.sh", "collect.sh"), ("fleet/payload.tar.gz", "payload.tar.gz")):
        if reuse is not None and dst == "payload.tar.gz" and st.get("skip_payload"):
            continue
        rc, out, err = fs("vm", "scp", src, "%s:/home/ubuntu/%s" % (vid, dst), timeout=240, retries=2)
        if rc != 0:
            log("%s scp %s failed: %s" % (slug, dst, (err or out).strip()[-200:]), "error")
            st["shards"][vid]["phase"] = "upload-failed"; save_state(st); return
    st["shards"][vid]["phase"] = "installing"; save_state(st)
    args = ["vm", "exec", vid, "--timeout-ms", "280000", "--", "bash", "/home/ubuntu/start.sh",
            "s%d" % i, "n%d" % n, condition, "k%d" % seeds] + ([tag] if tag else [])  # CLI chokes on bare numeric args
    rc, out, err = fs(*args, timeout=300, retries=1)
    ok = rc == 0 and "started" in out
    with LOCK:
        st["shards"][vid]["phase"] = "running" if ok else "start-failed"
        save_state(st)
    log("%s: %s" % (slug, out.strip().splitlines()[-1] if ok else "start failed: " + (err or out).strip()[-200:]),
        "ok" if ok else "error")


def do_launch(n, condition, seeds, tag, reuse=False):
    BUSY["launch"] = True
    try:
        log("launch: %d VMs, condition=%s, seeds=%d%s" % (n, condition, seeds, " tag=" + tag if tag else ""))
        old = load_state()
        st = {"launched_at": time.time(), "n_vms": n, "condition": condition, "seeds": seeds, "tag": tag, "shards": {}}
        existing = None
        if reuse:   # restart shards on already-created nr-fleet-* VMs (payload already uploaded)
            existing = {v.get("slug"): v["id"] for v in fleet_vms()}
            st["launched_at"] = old.get("launched_at") or st["launched_at"]
            st["skip_payload"] = True
            log("reusing %d existing VMs" % len(existing))
        else:
            build_payload()
        save_state(st)
        list(POOL.map(lambda i: launch_one(i, n, condition, seeds, tag, st, existing), range(n)))
        good = sum(1 for s in st["shards"].values() if s["phase"] == "running")
        log("launch finished: %d/%d shards running" % (good, n), "ok" if good == n else "warn")
    except Exception as e:
        log("launch error: %r" % e, "error")
    finally:
        BUSY["launch"] = False


def fleet_vms(prefix="nr-fleet-"):
    rc, out, err = fs("vm", "list", "--output", "json", timeout=60)
    return [v for v in parse_json(out).get("vms", []) if str(v.get("slug") or "").startswith(prefix)] if rc == 0 else []


def collect_one(v):
    vid, slug = v["id"], v.get("slug")
    rc, out, err = fs("vm", "exec", vid, "--", "bash", "/home/ubuntu/collect.sh", timeout=120)
    if rc != 0:
        log("%s collect tar failed: %s" % (slug, (err or out)[-150:]), "error"); return 0
    local = "fleet/results_%s.tgz" % slug
    rc, out, err = fs("vm", "scp", "%s:/home/ubuntu/results.tgz" % vid, local, timeout=120)
    if rc != 0:
        log("%s collect scp failed: %s" % (slug, (err or out)[-150:]), "error"); return 0
    dest = ROOT / "neurorover" / "results"; dest.mkdir(parents=True, exist_ok=True)
    added = 0
    with tarfile.open(ROOT / local) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if not m.isfile() or not name or (dest / name).exists():
                continue
            (dest / name).write_bytes(tf.extractfile(m).read()); added += 1
    log("%s: collected %d new result files" % (slug, added), "ok")
    return added


def do_collect():
    vms = fleet_vms()
    log("collect: %d VMs" % len(vms))
    total = sum(POOL.map(collect_one, vms))
    log("collect finished: %d new files in neurorover/results" % total, "ok")


def do_teardown():
    vms = fleet_vms()
    log("teardown: deleting %d VMs" % len(vms), "warn")

    def kill(v):
        rc, out, err = fs("vm", "delete", v["id"], timeout=90)
        log("%s %s" % (v.get("slug"), "deleted" if rc == 0 else "delete failed: " + (err or out)[-150:]),
            "ok" if rc == 0 else "error")
    list(POOL.map(kill, vms))
    log("teardown finished", "ok")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self.send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/api/vms":
            with LOCK:
                self.send(200, dict(CACHE, busy=BUSY["launch"], now=time.time()))
        elif p == "/api/events":
            self.send(200, {"events": list(EVENTS)[-150:]})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except Exception:
            return self.send(400, {"error": "bad json"})
        if p == "/api/launch":
            if BUSY["launch"]:
                return self.send(409, {"error": "launch already in progress"})
            try:
                nv = int(body.get("n_vms", 10)); seeds = int(body.get("seeds", 10))
                cond = str(body.get("condition", "culture_teach")); tag = str(body.get("tag") or "")
            except Exception:
                return self.send(400, {"error": "bad params"})
            if not (1 <= nv <= 10) or seeds < 1 or not cond.replace("_", "").isalnum() \
                    or (tag and not tag.replace("_", "").isalnum()):
                return self.send(400, {"error": "n_vms must be 1..10; condition/tag alphanumeric"})
            threading.Thread(target=do_launch, args=(nv, cond, seeds, tag, bool(body.get("reuse"))), daemon=True).start()
            self.send(202, {"ok": True, "n_vms": nv, "condition": cond, "seeds": seeds})
        elif p == "/api/collect":
            threading.Thread(target=do_collect, daemon=True).start(); self.send(202, {"ok": True})
        elif p == "/api/teardown":
            threading.Thread(target=do_teardown, daemon=True).start(); self.send(202, {"ok": True})
        else:
            self.send(404, {"error": "not found"})


if __name__ == "__main__":
    threading.Thread(target=refresher, daemon=True).start()
    log("fleet server on http://127.0.0.1:8770 (cli: %s)" % NPX)
    ThreadingHTTPServer(("127.0.0.1", 8770), H).serve_forever()
