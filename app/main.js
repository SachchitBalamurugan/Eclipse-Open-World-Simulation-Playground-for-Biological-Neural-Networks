// NeuroRover Sim - Electron main process.
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const fsp = fs.promises;
const { spawn, execFile } = require('child_process');

const REPO = path.resolve(__dirname, '..');
const RUNS = path.join(REPO, 'runs');
const RESULTS = path.join(REPO, 'neurorover', 'results');
const SHOT = process.argv.includes('--screenshot');
const SELFTEST = process.argv.includes('--selftest');
const argVal = (k) => { const a = process.argv.find((x) => x.startsWith('--' + k + '=')); return a ? a.slice(k.length + 3) : ''; };
const SHOT_OUT = argVal('out');
let win = null;
const children = new Set();

function pythonExe() {
  const cands = [path.join(REPO, '.venv', 'Scripts', 'python.exe'), path.join(REPO, '.venv', 'bin', 'python')];
  for (const c of cands) if (fs.existsSync(c)) return c;
  return 'python';
}
const safeName = (s) => typeof s === 'string' && /^[A-Za-z0-9_.-]+$/.test(s) && !s.includes('..');
async function readText(p) { try { return await fsp.readFile(p, 'utf8'); } catch (e) { return null; } }
function parseCsv(text) {
  if (!text) return { header: [], rows: [] };
  const lines = text.split(/\r?\n/).filter((l) => l.length);
  if (!lines.length) return { header: [], rows: [] };
  return { header: lines[0].split(','), rows: lines.slice(1).map((l) => l.split(',')) };
}
function splitRunName(name) {
  const m = /^(.*)_s(\d+)$/.exec(name);
  return m ? { condition: m[1], seed: Number(m[2]) } : { condition: name, seed: null };
}
const countRows = (txt) => (txt ? Math.max(0, txt.split('\n').filter((l) => l.trim()).length - 1) : 0);

async function listRuns() {
  let dirs = [];
  try { dirs = (await fsp.readdir(RUNS, { withFileTypes: true })).filter((d) => d.isDirectory()); } catch (e) { return []; }
  const out = [];
  for (const d of dirs) {
    const dir = path.join(RUNS, d.name);
    const trialsTxt = await readText(path.join(dir, 'trials.csv'));
    if (trialsTxt == null) continue;
    let meta = null;
    try { meta = JSON.parse(await readText(path.join(dir, 'meta.json'))); } catch (e) { meta = null; }
    const guess = splitRunName(d.name);
    let mtime = 0;
    try { mtime = (await fsp.stat(path.join(dir, 'trials.csv'))).mtimeMs; } catch (e) {}
    out.push({
      name: d.name,
      condition: (meta && meta.condition) || guess.condition,
      seed: meta && meta.seed != null ? meta.seed : guess.seed,
      final_auc: meta && typeof meta.final_auc === 'number' ? meta.final_auc : null,
      nTrials: countRows(trialsTxt), complete: !!meta, mtime,
    });
  }
  out.sort((a, b) => b.mtime - a.mtime);
  return out;
}

// Events are written in time order, same order as trials.csv rows. Eval trial indices repeat at
// every checkpoint, so segment by walking in order.
function segmentEvents(text, trialRows, withKind) {
  const n = trialRows.length;
  const offsets = new Uint32Array(n + 1);
  if (!text || !n) return { t: new Uint16Array(0), ch: new Uint8Array(0), kind: new Uint8Array(0), offsets };
  const lines = text.split('\n');
  const cap = lines.length;
  const t = new Uint16Array(cap), ch = new Uint8Array(cap), kind = new Uint8Array(withKind ? cap : 0);
  const keyOf = (r) => r.trial + ',' + r.phase;
  let k = 0, seg = 0, curKey = keyOf(trialRows[0]);
  for (let i = 1; i < lines.length; i++) {
    const p = lines[i].trim().split(',');
    if (p.length < 4) continue;
    const key = p[0] + ',' + p[1];
    if (key !== curKey) {
      let j = seg + 1;
      while (j < n && keyOf(trialRows[j]) !== key) j++;
      if (j >= n) continue;
      for (let s = seg + 1; s <= j; s++) offsets[s] = k;
      seg = j; curKey = key;
    }
    const tv = Number(p[2]), cv = Number(p[3]);
    if (!(tv >= 0) || !(cv >= 0 && cv < 64)) continue;
    t[k] = Math.min(65535, tv); ch[k] = cv;
    if (withKind) {
      const kd = p[4] || '';
      // 0 sensory, 1 feedback, 2 reward, 3 motor teaching
      let code = kd === 'sensory' ? 0 : kd === 'reward' ? 2 : 1;
      if (cv >= 40 && cv < 56) code = 3;
      kind[k] = code;
    }
    k++;
  }
  for (let s = seg + 1; s <= n; s++) offsets[s] = k;
  return { t: t.slice(0, k), ch: ch.slice(0, k), kind: withKind ? kind.slice(0, k) : kind, offsets };
}

async function loadRun(name) {
  if (!safeName(name)) throw new Error('bad run name');
  const dir = path.join(RUNS, name);
  const [trialsTxt, spikesTxt, stimsTxt, weightsTxt, metaTxt] = await Promise.all(
    ['trials.csv', 'spikes.csv', 'stims.csv', 'weights.csv', 'meta.json'].map((f) => readText(path.join(dir, f))));
  if (trialsTxt == null) throw new Error('trials.csv missing in ' + name);
  const tc = parseCsv(trialsTxt);
  const col = (n) => tc.header.indexOf(n);
  const num = (r, n, d) => { const i = col(n); const v = i >= 0 ? Number(r[i]) : NaN; return Number.isFinite(v) ? v : d; };
  const trials = tc.rows.filter((r) => r.length >= 2).map((r) => ({
    trial: String(r[col('trial')] || '0').trim(), phase: String(r[col('phase')] || 'train').trim(),
    label: num(r, 'station_label', -1), action: num(r, 'action', -1), correct: num(r, 'correct', -1),
    score: num(r, 'score', 0), deploy: num(r, 'deploy_spikes', 0), hold: num(r, 'hold_spikes', 0), total: num(r, 'total_spikes', 0),
  }));
  const spikes = segmentEvents(spikesTxt, trials, false);
  const stims = segmentEvents(stimsTxt, trials, true);
  const wc = parseCsv(weightsTxt);
  const weights = weightsTxt ? { header: wc.header, rows: wc.rows.map((r) => r.map(Number)) } : null;
  let meta = null; try { meta = JSON.parse(metaTxt); } catch (e) {}
  const guess = splitRunName(name);
  return { name, meta, condition: (meta && meta.condition) || guess.condition, seed: meta ? meta.seed : guess.seed, trials, spikes, stims, weights };
}

async function loadBenchmarks() {
  let files = [];
  try { files = (await fsp.readdir(RESULTS)).filter((f) => f.endsWith('.json')); } catch (e) { return { headline: [], variants: [], budgets: [], nFiles: 0 }; }
  const groups = new Map();
  let ok = 0;
  await Promise.all(files.map(async (f) => {
    try {
      const d = JSON.parse(await fsp.readFile(path.join(RESULTS, f), 'utf8'));
      if (!d || typeof d.condition !== 'string' || !d.auc || typeof d.auc !== 'object') return;
      const tag = typeof d.tag === 'string' ? d.tag : '';
      const key = d.condition + '|' + tag;
      if (!groups.has(key)) groups.set(key, { condition: d.condition, tag, per: {}, seeds: new Set() });
      const g = groups.get(key);
      g.seeds.add(d.seed);
      for (const [b, v] of Object.entries(d.auc)) if (typeof v === 'number' && Number.isFinite(v)) (g.per[b] = g.per[b] || []).push(v);
      ok++;
    } catch (e) {}
  }));
  const budgets = new Set();
  const rows = [...groups.values()].map((g) => {
    const stats = {};
    for (const [b, arr] of Object.entries(g.per)) {
      budgets.add(Number(b));
      const n = arr.length, mean = arr.reduce((a, c) => a + c, 0) / n;
      const sd = n > 1 ? Math.sqrt(arr.reduce((a, c) => a + (c - mean) ** 2, 0) / (n - 1)) : 0;
      stats[b] = { mean, sem: n > 1 ? sd / Math.sqrt(n) : null, n };
    }
    return { condition: g.condition, tag: g.tag, nSeeds: g.seeds.size, stats };
  });
  rows.sort((a, b) => a.condition.localeCompare(b.condition) || a.tag.localeCompare(b.tag));
  return { headline: rows.filter((r) => !r.tag), variants: rows.filter((r) => r.tag),
    budgets: [...budgets].filter(Number.isFinite).sort((a, b) => a - b), nFiles: ok };
}

function runnableConditions() {
  return new Promise((resolve) => {
    execFile(pythonExe(), ['-c', 'import json;from neurorover import bench;print(json.dumps(list(bench.CULTURE_CONDS)))'],
      { cwd: REPO, timeout: 20000, windowsHide: true }, (err, stdout) => {
        if (err) return resolve(null);
        try { const a = JSON.parse(String(stdout).trim().split(/\r?\n/).pop()); resolve(Array.isArray(a) ? a.map(String) : null); } catch (e) { resolve(null); }
      });
  });
}

function send(chan, payload) { if (win && !win.isDestroyed()) win.webContents.send(chan, payload); }

function startRun(opts) {
  const condition = String(opts.condition || '').trim();
  const seed = Math.trunc(Number(opts.seed)), budget = Math.trunc(Number(opts.budget)), ev = Math.trunc(Number(opts.eval));
  if (!/^[A-Za-z0-9_-]+$/.test(condition)) throw new Error('invalid condition name');
  if (!(seed >= 0) || !(budget >= 1) || !(ev >= 2)) throw new Error('need seed>=0, budget>=1, eval>=2');
  const id = condition + '_s' + seed;
  const args = ['-u', '-m', 'neurorover.record_run', '--condition', condition, '--seed', String(seed), '--budget', String(budget), '--eval', String(ev)];
  const child = spawn(pythonExe(), args, { cwd: REPO, windowsHide: true, env: Object.assign({}, process.env, { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' }) });
  children.add(child);
  send('run-log', { id, stream: 'sys', line: '$ python ' + args.slice(1).join(' ') });
  const pipe = (stream, tag) => {
    let buf = '';
    stream.setEncoding('utf8');
    stream.on('data', (d) => {
      buf += d;
      const parts = buf.split(/\r?\n|\r/); buf = parts.pop();
      for (const line of parts) if (line.length) send('run-log', { id, stream: tag, line });
    });
    stream.on('end', () => { if (buf.length) send('run-log', { id, stream: tag, line: buf }); });
  };
  pipe(child.stdout, 'out'); pipe(child.stderr, 'err');
  // record_run prints only a final line, so also report progress from trials.csv while it runs.
  const t0 = Date.now();
  const hb = setInterval(async () => {
    const n = countRows(await readText(path.join(RUNS, id, 'trials.csv')));
    send('run-log', { id, stream: 'sys', line: '... running ' + ((Date.now() - t0) / 1000).toFixed(0) + 's, ' + n + ' trial rows on disk' });
  }, 3000);
  child.on('error', (e) => { clearInterval(hb); children.delete(child); send('run-log', { id, stream: 'err', line: String(e) }); send('run-done', { id, code: -1 }); });
  child.on('close', (code) => { clearInterval(hb); children.delete(child); send('run-done', { id, code, wall_s: (Date.now() - t0) / 1000 }); });
  return { id };
}

async function listLogs() {
  try { return (await fsp.readdir(REPO)).filter((f) => f.toLowerCase().endsWith('.log')).sort(); } catch (e) { return []; }
}
async function tailLog(name, fromPos) {
  if (!safeName(name) || !name.toLowerCase().endsWith('.log')) throw new Error('bad log name');
  const p = path.join(REPO, name);
  const st = await fsp.stat(p);
  const fresh = !(typeof fromPos === 'number' && fromPos >= 0 && fromPos <= st.size);
  const start = fresh ? Math.max(0, st.size - 5000) : fromPos;
  const len = st.size - start;
  if (len <= 0) return { pos: st.size, lines: [] };
  const fh = await fsp.open(p, 'r');
  try {
    const buf = Buffer.alloc(len);
    await fh.read(buf, 0, len, start);
    const lines = buf.toString('utf8').split(/\r?\n/);
    if (fresh && start > 0) lines.shift();
    return { pos: st.size, lines: lines.filter((l) => l.length) };
  } finally { await fh.close(); }
}

async function capture() {
  try {
    const img = await win.webContents.capturePage();
    const out = SHOT_OUT && /^[A-Za-z0-9_.-]+\.png$/.test(SHOT_OUT) ? SHOT_OUT : 'screenshot.png';
    await fsp.writeFile(path.join(__dirname, out), img.toPNG());
  } catch (e) { console.error('capture failed', e); }
}

const handle = (chan, fn) => ipcMain.handle(chan, async (_e, ...a) => {
  try { return { ok: true, data: await fn(...a) }; } catch (e) { return { ok: false, error: String((e && e.message) || e) }; }
});
handle('list-runs', listRuns);
handle('load-run', loadRun);
handle('benchmarks', loadBenchmarks);
handle('conditions', runnableConditions);
handle('start-run', startRun);
handle('list-logs', listLogs);
handle('tail-log', tailLog);
// ---------- assets + replay files (sim/runs/*/trajectory*.json, surgical/results/trajectory.json) ----------
const ASSETS = path.join(__dirname, 'assets');
const SIM_RUNS = path.join(REPO, 'sim', 'runs');
const SURG_RESULTS = path.join(REPO, 'surgical', 'results');
const PROPS_XML = path.join(REPO, 'assets', 'simready', 'mjcf', 'props.xml');
async function readAsset(rel) {
  if (rel === 'simready/props.xml') { // canonical layout lives in assets/simready/mjcf; app copy is the fallback
    try { return await fsp.readFile(PROPS_XML); } catch (e) {}
  }
  if (typeof rel !== 'string' || rel.includes('..') || !/^[A-Za-z0-9_./-]+$/.test(rel)) throw new Error('bad asset path');
  return fsp.readFile(path.join(ASSETS, rel));
}
async function listReplays() {
  const rover = [];
  let dirs = [];
  try { dirs = (await fsp.readdir(SIM_RUNS, { withFileTypes: true })).filter((d) => d.isDirectory()); } catch (e) {}
  for (const d of dirs) {
    let files = [];
    try { files = (await fsp.readdir(path.join(SIM_RUNS, d.name))).filter((f) => /^trajectory.*\.json$/.test(f)); } catch (e) {}
    for (const f of files) {
      try { const st = await fsp.stat(path.join(SIM_RUNS, d.name, f)); rover.push({ run: d.name, file: f, mtime: st.mtimeMs, size: st.size }); } catch (e) {}
    }
  }
  // newest first, but prefer the canonical trajectory.json of a run over smoke tests
  rover.sort((a, b) => (b.file === 'trajectory.json') - (a.file === 'trajectory.json') || b.mtime - a.mtime);
  let surgical = null;
  try { const st = await fsp.stat(path.join(SURG_RESULTS, 'trajectory.json')); surgical = { file: 'surgical/results/trajectory.json', mtime: st.mtimeMs, size: st.size }; } catch (e) {}
  return { rover, surgical };
}
async function loadReplay(kind, run, file) {
  let p;
  if (kind === 'surgical') p = path.join(SURG_RESULTS, 'trajectory.json');
  else if (kind === 'rover' && safeName(run) && safeName(file) && /^trajectory.*\.json$/.test(file)) p = path.join(SIM_RUNS, run, file);
  else throw new Error('bad replay request');
  // Python's json writes NaN / Infinity, which JSON.parse rejects: map them to null
  return JSON.parse((await fsp.readFile(p, 'utf8')).replace(/(?<!["\w])-?(NaN|Infinity)(?!["\w])/g, 'null'));
}
const watchers = [];
let replayTimer = null;
function watchReplays() {
  const ping = (fn) => { if (!/trajectory.*\.json$/.test(String(fn || 'trajectory.json'))) return; clearTimeout(replayTimer); replayTimer = setTimeout(() => send('replays-changed', {}), 1500); };
  for (const dir of [SIM_RUNS, SURG_RESULTS]) {
    try { watchers.push(fs.watch(dir, { recursive: true }, (_ev, fn) => ping(fn))); } catch (e) {
      // directory may not exist yet: poll for it
      const t = setInterval(() => { if (fs.existsSync(dir)) { clearInterval(t); try { watchers.push(fs.watch(dir, { recursive: true }, (_ev, fn) => ping(fn))); ping(); } catch (e2) {} } }, 5000);
    }
  }
}
handle('read-asset', readAsset);
handle('list-replays', listReplays);
handle('load-replay', loadReplay);
handle('selftest-report', async (report) => {
  await fsp.writeFile(path.join(__dirname, 'selftest.json'), JSON.stringify(report, null, 2));
  if (SELFTEST) { await capture(); setTimeout(() => app.quit(), 200); }
  return true;
});

// ---------- fleet: proxy to fleet/server.py on :8770 (started as a child if not answering), or a mock feed ----------
const http = require('http');
const MOCK_FLEET = process.argv.includes('--mock-fleet');
const FLEET = { host: '127.0.0.1', port: Number(process.env.NR_FLEET_PORT) || 8770 };
let fleetChild = null;
function fleetReq(method, p, body, timeout) {
  return new Promise((resolve, reject) => {
    const data = body ? Buffer.from(JSON.stringify(body)) : null;
    const req = http.request({ host: FLEET.host, port: FLEET.port, path: p, method, timeout: timeout || 4000,
      headers: data ? { 'Content-Type': 'application/json', 'Content-Length': data.length } : {} }, (res) => {
      let buf = ''; res.setEncoding('utf8'); res.on('data', (d) => buf += d);
      res.on('end', () => { try { resolve(Object.assign({ status: res.statusCode }, JSON.parse(buf || '{}'))); } catch (e) { reject(new Error('bad json from fleet server')); } });
    });
    req.on('timeout', () => req.destroy(new Error('fleet server timeout')));
    req.on('error', reject);
    if (data) req.write(data); req.end();
  });
}
async function ensureFleetServer() {
  if (MOCK_FLEET || FLEET.port !== 8770) return;
  try { await fleetReq('GET', '/api/events', null, 1500); return; } catch (e) {}
  const py = pythonExe();
  fleetChild = spawn(py, ['-u', path.join('fleet', 'server.py')], { cwd: REPO, windowsHide: true, env: Object.assign({}, process.env, { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' }) });
  fleetChild.stdout.on('data', () => {}); fleetChild.stderr.on('data', () => {});
  fleetChild.on('close', () => { fleetChild = null; });
}
function stopFleetServer() { if (fleetChild) { try { fleetChild.kill(); } catch (e) {} fleetChild = null; } }

// mock: 10 VMs with synthetic bench logs, clearly flagged mock:true
const mock = { t0: Date.now(), vms: [] };
function mockFleet() {
  const t = (Date.now() - mock.t0) / 1000;
  if (!mock.vms.length) for (let i = 0; i < 10; i++) mock.vms.push({ id: 'mock-' + i, slug: 'nr-fleet-' + i, seeds: [], log: [], next: 2 + i * 0.7, load: [3.6, 3.4, 3.0], skill: 0.08 + 0.02 * (i % 4) });
  const vms = mock.vms.map((m, i) => {
    while (t > m.next && m.seeds.length < 5) {
      const seed = i * 5 + m.seeds.length, auc = {};
      [25, 50, 100, 200].forEach((b, k) => { auc[b] = Math.min(0.98, 0.52 + m.skill * (k + 1) + (Math.sin(seed * 7.1 + k) * 0.03)); });
      m.seeds.push(seed);
      m.log.push(`culture_teach s${seed}: ` + Object.entries(auc).map(([b, v]) => b + ':' + v.toFixed(3)).join(' ') + `  (${(40 + (seed % 7) * 3).toFixed(1)}s)`);
      m.next += 3 + (i % 3);
    }
    if (Math.random() < 0.4) m.log.push(`[bench] shard ${i}/10 seed ${i * 5 + m.seeds.length} trial ${Math.floor(t * 7 % 400)} eval checkpoint`);
    m.log = m.log.slice(-5);
    const l = 3.2 + Math.sin(t / 5 + i) * 0.6 + Math.random() * 0.3;
    return { id: m.id, slug: m.slug, state: 'running', resources: { vcpus: 4, memory_mb: 8192 },
      probe: { ok: true, results: m.seeds.length * 4, procs: 4, nproc: 4, load: [l, l * 0.9, l * 0.8], mem_used_mb: 2100 + i * 90 + Math.round(Math.random() * 60), mem_total_mb: 7940, log: m.log.slice(), ts: Date.now() / 1000 },
      shard: i, nshards: 10, expected: 20, phase: 'bench' };
  });
  return { mock: true, serverUp: false, vms, runningCount: 10, launch: { condition: 'culture_teach', seeds: 50, n_vms: 10 }, price_per_vm_hour: 0.27,
    events: [{ t: Date.now() / 1000, level: 'info', msg: 'MOCK DATA: synthetic fleet, no VMs exist' }] };
}
async function fleetGet() {
  if (MOCK_FLEET) return mockFleet();
  let vmsRes, ev;
  try { vmsRes = await fleetReq('GET', '/api/vms'); } catch (e) { return { serverUp: false, vms: [], error: String(e.message || e) }; }
  try { ev = await fleetReq('GET', '/api/events'); } catch (e) { ev = { events: [] }; }
  return Object.assign(vmsRes, { serverUp: true, events: ev.events || [] });
}
async function fleetPost(p, body) {
  if (!['/api/launch', '/api/collect', '/api/teardown'].includes(p)) throw new Error('bad fleet endpoint');
  if (MOCK_FLEET) return { ok: true, mock: true };
  return fleetReq('POST', p, body || {}, 8000);
}
const popouts = new Map();
function fleetPopout(ids) {
  const { screen } = require('electron');
  ids = (Array.isArray(ids) ? ids : [ids]).map(String).filter((s) => /^[A-Za-z0-9_.:-]+$/.test(s)).slice(0, 10);
  const wa = screen.getPrimaryDisplay().workArea, n = ids.length;
  const cols = n > 1 ? Math.ceil(Math.sqrt(n * wa.width / wa.height)) : 1, rows = Math.ceil(n / cols);
  ids.forEach((id, k) => {
    const old = popouts.get(id); if (old && !old.isDestroyed()) old.close();
    const w = n > 1 ? Math.floor(wa.width / cols) : 720, h = n > 1 ? Math.floor(wa.height / rows) : 560;
    const bw = new BrowserWindow({ x: n > 1 ? wa.x + (k % cols) * w : undefined, y: n > 1 ? wa.y + Math.floor(k / cols) * h : undefined, width: w, height: h,
      backgroundColor: '#000000', title: 'NeuroRover Sim - ' + id, autoHideMenuBar: true,
      webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false } });
    bw.loadFile(path.join(__dirname, 'popout.html'), { query: { vm: id } });
    popouts.set(id, bw);
  });
  return ids.length;
}
handle('fleet-get', fleetGet);
handle('fleet-post', fleetPost);
handle('fleet-popout', fleetPopout);

// ---------- BNN Robot Studio: thin wrapper over `python -m bnn_platform studio ...` (docs/STUDIO_CONTRACT.md) ----------
// User strings only ever reach Python as separate argv entries (no shell), after validation.
const ROBOT_RE = /^[a-z0-9_]{1,40}$/;
const PY_ENV = () => Object.assign({}, process.env, { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' });
let studioChild = null; // the 'main' slot (local train / replay / transfer run)
let fleetLaunch = null;  // the 'fleet' slot: fleet launches run beside a replay
const DEMO = process.argv.includes('--demo');
const DEMO_TITLE = 'BNN Robot Studio Demo';
const demoVms = new Set(); // VMs created by fleet launches in this session (demo teardown only touches these)
function lastJson(text) {
  const lines = String(text || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) { if (lines[i][0] !== '{') continue; try { return JSON.parse(lines[i]); } catch (e) {} }
  return null;
}
function errTail(stderr) {
  const l = String(stderr || '').split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  return l.slice(-3).join(' | ').slice(0, 600);
}
// one-shot studio command: resolve with its JSON, reject with the CLI's own error message
function studioCli(args, timeout, mod) {
  const pre = mod === 'transfer' ? ['-u', '-m', 'bnn_platform.transfer'] : ['-u', '-m', 'bnn_platform', 'studio'];
  return new Promise((resolve, reject) => {
    execFile(pythonExe(), [...pre, ...args],
      { cwd: REPO, timeout: timeout || 60000, windowsHide: true, maxBuffer: 256 * 1024 * 1024, env: PY_ENV() }, (err, stdout, stderr) => {
        const j = lastJson(stdout);
        if (j && j.ok === false) return reject(new Error(j.error || 'studio command failed'));
        if (err) return reject(new Error((j && j.error) || errTail(stderr) || String(err.message || err)));
        if (!j) return reject(new Error('no JSON from `' + pre.slice(2).join(' ') + ' ' + args[0] + '`' + (stderr ? ': ' + errTail(stderr) : '')));
        resolve(j);
      });
  });
}
async function studioPickFile() {
  const { dialog } = require('electron');
  const r = await dialog.showOpenDialog(win, { title: 'Import robot', properties: ['openFile'],
    filters: [{ name: 'Robot description (MJCF, URDF, zip)', extensions: ['xml', 'urdf', 'zip'] }] });
  return r.canceled || !r.filePaths.length ? null : r.filePaths[0];
}
async function studioImport(file, opts) {
  opts = opts || {};
  if (typeof file !== 'string' || !path.isAbsolute(file) || file.length > 1024) throw new Error('import needs an absolute file path');
  if (!/\.(xml|urdf|zip)$/i.test(file)) throw new Error('unsupported file type: use .xml (MJCF), .urdf or .zip');
  let st; try { st = await fsp.stat(file); } catch (e) { throw new Error('file not found: ' + file); }
  if (!st.isFile()) throw new Error('not a file: ' + file);
  if (st.size > 512 * 1024 * 1024) throw new Error('file too large (>512 MB)');
  const args = ['import', '--file', file];
  if (opts.name != null && opts.name !== '') { if (!ROBOT_RE.test(opts.name)) throw new Error('robot name must match [a-z0-9_]{1,40}'); args.push('--name', opts.name); }
  if (opts.ee != null && opts.ee !== '') { if (!/^[A-Za-z0-9_.:-]{1,80}$/.test(opts.ee)) throw new Error('bad end-effector name'); args.push('--ee', opts.ee); }
  return studioCli(args, 180000);
}
async function studioScene(name) {
  if (!ROBOT_RE.test(String(name))) throw new Error('bad robot name');
  return studioCli(['scene', '--robot', name], 120000);
}
function intIn(v, lo, hi, what) { const n = Number(v); if (!Number.isInteger(n) || n < lo || n > hi) throw new Error(what + ' must be an integer in [' + lo + ', ' + hi + ']'); return n; }
function studioTrain(cfg) {
  cfg = cfg || {};
  const robot = String(cfg.robot || '');
  if (!ROBOT_RE.test(robot)) throw new Error('bad robot name');
  const task = cfg.task || 'reach'; if (!['reach'].includes(task)) throw new Error('unknown task');
  const platform = cfg.platform || 'local'; if (!['local', 'fleet'].includes(platform)) throw new Error('platform must be local or fleet');
  const rule = cfg.rule || 'teach'; if (!['teach', 'frozen'].includes(rule)) throw new Error('rule must be teach or frozen');
  const episodes = intIn(cfg.episodes ?? 20, 1, 1000, 'episodes');
  const neurons = intIn(cfg.neurons ?? 1024, 64, 4096, 'neurons');
  const electrodes = intIn(cfg.electrodes ?? 64, 4, 1024, 'electrodes');
  const seed = intIn(cfg.seed ?? 0, 0, 1000000, 'seed');
  const vms = intIn(cfg.vms ?? 4, 1, 32, 'vms');
  const lr = Number(cfg.lr ?? 0.02); if (!(Number.isFinite(lr) && lr > 0 && lr <= 1)) throw new Error('learning rate must be in (0, 1]');
  const args = ['-u', '-m', 'bnn_platform', 'studio', 'train', '--robot', robot, '--task', task, '--platform', platform,
    '--episodes', String(episodes), '--lr', String(lr), '--neurons', String(neurons), '--electrodes', String(electrodes),
    '--rule', rule, '--seed', String(seed)];
  if (platform === 'fleet') { args.push('--vms', String(vms)); if (cfg.dryRun) args.push('--dry-run'); }
  const slot = platform === 'fleet' ? 'fleet' : 'main';
  if (slot === 'main' ? studioChild : fleetLaunch) throw new Error('a ' + (slot === 'fleet' ? 'fleet launch' : 'run') + ' is already active - stop it first');
  return spawnStream(args, 'training', slot);
}
// spawn a JSON-Lines producing python command; forward each stdout line as a studio-event (partial lines buffered)
function spawnStream(args, what, slot) {
  slot = slot || 'main';
  const child = spawn(pythonExe(), args, { cwd: REPO, windowsHide: true, env: PY_ENV() });
  if (slot === 'fleet') fleetLaunch = child; else studioChild = child;
  children.add(child);
  const ev = (o) => {
    if (o.type === 'fleet' && Array.isArray(o.jobs) && !o.dry_run) for (const j of o.jobs) if (j && typeof j.vm === 'string' && /^bnn-/.test(String(j.slug || ''))) demoVms.add(j.vm);
    send('studio-event', Object.assign(o, { slot }));
  };
  ev({ type: 'log', level: 'sys', message: '$ python ' + args.slice(1).join(' ') });
  let sawEnd = false, buf = '', ebuf = '';
  const onLine = (line) => {
    const s = line.trim(); if (!s) return;
    let o = null; try { o = JSON.parse(s); } catch (e) {}
    if (o && typeof o === 'object' && typeof o.type === 'string') { if (o.type === 'done' || o.type === 'error') sawEnd = true; ev(o); }
    else ev({ type: 'log', level: 'out', message: s });
  };
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (d) => { buf += d; const parts = buf.split(/\r?\n/); buf = parts.pop(); parts.forEach(onLine); });
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (d) => {
    ebuf += d; const parts = ebuf.split(/\r?\n|\r/); ebuf = parts.pop();
    for (const line of parts) if (line.trim()) ev({ type: 'log', level: 'err', message: line });
  });
  const finish = (code, err) => {
    if (slot === 'fleet') { if (fleetLaunch !== child) return; fleetLaunch = null; }
    else { if (studioChild !== child) return; studioChild = null; }
    children.delete(child);
    if (buf.trim()) onLine(buf);
    if (ebuf.trim()) ev({ type: 'log', level: 'err', message: ebuf });
    if (err) ev({ type: 'error', message: String(err.message || err) });
    else if (!sawEnd) ev({ type: 'error', message: child.killedByUser ? 'stopped by user' : what + ' process exited with code ' + code });
    ev({ type: 'exit', code, stopped: !!child.killedByUser });
  };
  child.on('error', (e) => finish(-1, e));
  child.on('close', (code) => finish(code, null));
  return { pid: child.pid, args: args.slice(1) };
}
function killTree(child) {
  if (!child || child.exitCode != null) return;
  if (process.platform === 'win32') execFile('taskkill', ['/pid', String(child.pid), '/t', '/f'], { windowsHide: true }, () => {});
  else { try { child.kill('SIGTERM'); } catch (e) {} }
}
function studioStop(slot) {
  const c = slot === 'fleet' ? fleetLaunch : studioChild;
  if (!c) return { stopped: false };
  c.killedByUser = true; killTree(c);
  return { stopped: true };
}
handle('studio-list', () => studioCli(['list'], 90000));
handle('studio-pick-file', studioPickFile);
handle('studio-import', studioImport);
handle('studio-scene', studioScene);
handle('studio-train', studioTrain);
handle('studio-stop', studioStop);
handle('studio-runs', () => studioCli(['runs'], 60000));
handle('studio-fleet-status', () => studioCli(['fleet-status'], 60000));
const RUNID_RE = /^[A-Za-z0-9_.-]{1,160}$/;
function checkRunId(id) { if (typeof id !== 'string' || !RUNID_RE.test(id) || id.includes('..')) throw new Error('bad run id'); return id; }
handle('studio-pack', (id) => studioCli(['pack', '--run', checkRunId(id)], 180000));
// demo only: delete the Freestyle VMs that fleet launches in THIS session reported (slug bnn-*); never anything else
handle('studio-demo-teardown', async () => {
  if (!DEMO) throw new Error('teardown is only available in --demo mode');
  const out = [];
  for (const id of [...demoVms]) {
    if (!/^[A-Za-z0-9_-]{1,80}$/.test(id)) continue;
    const r = await new Promise((res) => execFile('cmd.exe', ['/d', '/c', 'npx.cmd', '-y', 'freestyle@latest', 'vm', 'delete', id],
      { cwd: REPO, timeout: 90000, windowsHide: true }, (err, so, se) => res({ vm: id, ok: !err, msg: String(err ? (se || err.message) : so).trim().slice(-200) })));
    if (r.ok) demoVms.delete(id);
    out.push(r);
  }
  return out;
});
handle('demo-quit', () => { setTimeout(() => app.quit(), 300); return true; });
handle('studio-replay', (id) => { if (studioChild) throw new Error('a run is already active - stop it first'); return spawnStream(['-u', '-m', 'bnn_platform', 'studio', 'replay', '--run', checkRunId(id)], 'replay'); });
handle('studio-download', async (id) => {
  checkRunId(id);
  const r = await studioCli(['pack', '--run', id], 180000);
  if (!r.zip) throw new Error('pack returned no zip');
  const src = path.resolve(REPO, r.zip);
  const { dialog } = require('electron');
  const s = await dialog.showSaveDialog(win, { title: 'Save run', defaultPath: path.basename(src), filters: [{ name: 'Zip archive', extensions: ['zip'] }] });
  if (s.canceled || !s.filePath) return { saved: null, zip: r.zip };
  await fsp.copyFile(src, s.filePath);
  return { saved: s.filePath, zip: r.zip };
});

// ---------- sim-to-real export (python -m bnn_platform.transfer, see contract) ----------
const USER_RUNS = path.join(REPO, 'user_runs');
const TARGET_RE = /^[a-z0-9_]{1,40}$/;
function insideRepoPath(p, base, what) {
  if (typeof p !== 'string' || !p || p.length > 1024 || p.includes('\u0000')) throw new Error('bad ' + what);
  const abs = path.resolve(REPO, p);
  const rel = path.relative(base, abs);
  if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) throw new Error(what + ' must be inside ' + path.relative(REPO, base));
  return abs;
}
async function transferExport(runDir, target) {
  let abs;
  if (runDir === 'surgical/results/culture_teach.pkl') abs = path.join(REPO, 'surgical', 'results', 'culture_teach.pkl');
  else abs = insideRepoPath(runDir, USER_RUNS, 'run directory');
  if (!fs.existsSync(abs)) throw new Error('run not found: ' + runDir);
  target = target || 'generic';
  if (!TARGET_RE.test(target)) throw new Error('bad target id');
  return studioCli(['export', '--run', abs, '--target', target], 180000, 'transfer');
}
function transferRun(bundle, episodes) {
  if (studioChild) throw new Error('a run is already active - stop it first');
  const abs = insideRepoPath(bundle, USER_RUNS, 'bundle');
  let st = null; try { st = fs.statSync(abs); } catch (e) {}
  if (!st || !st.isDirectory()) throw new Error('bundle directory not found: ' + bundle);
  const ep = intIn(episodes ?? 3, 1, 50, 'episodes');
  return spawnStream(['-u', '-m', 'bnn_platform.transfer', 'run', '--bundle', abs, '--backend', 'mock', '--episodes', String(ep)], 'closed-loop dry run');
}
function transferReveal(p) {
  const abs = insideRepoPath(p, REPO, 'path');
  if (!fs.existsSync(abs)) throw new Error('not found: ' + p);
  require('electron').shell.showItemInFolder(abs);
  return true;
}
handle('transfer-targets', () => studioCli(['targets'], 60000, 'transfer'));
handle('transfer-export', transferExport);
handle('transfer-run', transferRun);
handle('transfer-reveal', transferReveal);

function createWindow() {
  if (DEMO) return createDemoWindow();
  const sizeArg = process.argv.find((a) => a.startsWith('--size='));
  const wh = sizeArg ? sizeArg.slice(7).split('x').map(Number) : [1500, 900];
  win = new BrowserWindow({
    width: wh[0] || 1500, height: wh[1] || 900, minWidth: 1100, minHeight: 640, useContentSize: true,
    backgroundColor: '#0f0f0f', title: 'BNN Robot Studio - simulated culture model', autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, backgroundThrottling: false },
  });
  const debug = SHOT || SELFTEST;
  const clog = path.join(__dirname, 'console.log');
  if (debug) fs.writeFileSync(clog, '');
  win.webContents.on('console-message', (_e, level, message, line, source) => {
    if (debug) fs.appendFileSync(clog, '[' + level + '] ' + message + ' (' + path.basename(String(source)) + ':' + line + ')\n');
  });
  win.loadFile(path.join(__dirname, 'index.html'), { query: { selftest: SELFTEST ? '1' : '0', tab: argVal('tab') || 'studio', camera: argVal('camera') || 'overview', replay: process.argv.includes('--replay') ? '1' : '0', studiodemo: ROBOT_RE.test(argVal('studio-demo')) ? argVal('studio-demo') : '', studiosec: /^[a-z]{1,20}$/.test(argVal('studio-section')) ? argVal('studio-section') : '', studiodeploy: process.argv.includes('--studio-deploy') ? '1' : '0' } });
  if (SHOT) win.webContents.once('did-finish-load', () => setTimeout(async () => { await capture(); app.quit(); }, Math.max(3000, Math.min(120000, Number(argVal('delay')) || 9000))));
  if (SELFTEST) setTimeout(() => app.quit(), 420000);
  win.on('closed', () => app.quit());
}

app.whenReady().then(() => { ensureFleetServer().catch(() => {}); createWindow(); watchReplays(); });
app.on('before-quit', () => { stopFleetServer(); watchers.forEach((w) => { try { w.close(); } catch (e) {} }); });
function createDemoWindow() {
  // exactly 1920x1080 physical pixels at (0,0): divide by the display scale factor so gdigrab captures 1920x1080
  const { screen } = require('electron');
  const sf = screen.getPrimaryDisplay().scaleFactor || 1;
  win = new BrowserWindow({
    x: 0, y: 0, width: Math.round(1920 / sf), height: Math.round(1080 / sf), useContentSize: true, resizable: false, maximizable: false,
    frame: false, alwaysOnTop: true, autoHideMenuBar: true, backgroundColor: '#0f0f0f', title: DEMO_TITLE,
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, backgroundThrottling: false },
  });
  win.setMenuBarVisibility(false);
  win.on('page-title-updated', (e) => { e.preventDefault(); win.setTitle(DEMO_TITLE); });
  const dlog = path.join(__dirname, 'demo.log'); fs.writeFileSync(dlog, '');
  win.webContents.on('console-message', (_e, level, message) => fs.appendFileSync(dlog, new Date().toISOString().slice(11, 19) + ' [' + level + '] ' + message + '\n'));
  if (process.argv.includes('--demo-shots')) { let k = 0; const t = setInterval(async () => { try { const img = await win.webContents.capturePage(); await fsp.writeFile(path.join(__dirname, 'demo_shot_' + String(++k).padStart(2, '0') + '.png'), img.toPNG()); } catch (e) { clearInterval(t); } }, 8000); }
  win.loadFile(path.join(__dirname, 'index.html'), { query: { selftest: '0', tab: 'studio', demo: '1', dryfleet: process.argv.includes('--demo-dry-fleet') ? '1' : '0' } });
  win.on('closed', () => app.quit());
}
app.on('window-all-closed', () => { if (studioChild) killTree(studioChild); if (fleetLaunch) killTree(fleetLaunch); for (const c of children) { try { c.kill(); } catch (e) {} } stopFleetServer(); app.quit(); });
