// DNAforge Culture Lab - Electron main process.
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
    await fsp.writeFile(path.join(__dirname, 'screenshot.png'), img.toPNG());
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
handle('selftest-report', async (report) => {
  await fsp.writeFile(path.join(__dirname, 'selftest.json'), JSON.stringify(report, null, 2));
  if (SELFTEST) { await capture(); setTimeout(() => app.quit(), 200); }
  return true;
});

function createWindow() {
  const sizeArg = process.argv.find((a) => a.startsWith('--size='));
  const wh = sizeArg ? sizeArg.slice(7).split('x').map(Number) : [1500, 900];
  win = new BrowserWindow({
    width: wh[0] || 1500, height: wh[1] || 900, minWidth: 1100, minHeight: 640, useContentSize: true,
    backgroundColor: '#070a10', title: 'DNAforge Culture Lab - simulated culture model', autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, nodeIntegration: false, backgroundThrottling: false },
  });
  const debug = SHOT || SELFTEST;
  const clog = path.join(__dirname, 'console.log');
  if (debug) fs.writeFileSync(clog, '');
  win.webContents.on('console-message', (_e, level, message, line, source) => {
    if (debug) fs.appendFileSync(clog, '[' + level + '] ' + message + ' (' + path.basename(String(source)) + ':' + line + ')\n');
  });
  win.loadFile(path.join(__dirname, 'index.html'), { query: { selftest: SELFTEST ? '1' : '0' } });
  if (SHOT) win.webContents.once('did-finish-load', () => setTimeout(async () => { await capture(); app.quit(); }, 9000));
  if (SELFTEST) setTimeout(() => app.quit(), 200000);
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => { for (const c of children) { try { c.kill(); } catch (e) {} } app.quit(); });
