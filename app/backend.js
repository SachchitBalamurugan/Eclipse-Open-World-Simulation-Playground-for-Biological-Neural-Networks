// Eclipse backend resolver: where the Python backend source lives, which python runs it, and where user data goes.
//   dev:      repo root (one level above app/), python from <repo>/.venv, data in the repo (user_runs/, user_robots/)
//   packaged: source in <resources>/backend (electron-builder extraResources),
//             python from %LOCALAPPDATA%\Eclipse\venv (created on first run), data in %LOCALAPPDATA%\Eclipse\data
const path = require('path');
const fs = require('fs');
const { spawn, execFileSync } = require('child_process');

let electronApp = null;
try { electronApp = require('electron').app; } catch (e) {}
const PACKAGED = !!(electronApp && electronApp.isPackaged) || process.env.ECLIPSE_FORCE_PACKAGED === '1';

const LOCAL = process.env.LOCALAPPDATA || path.join(require('os').homedir(), 'AppData', 'Local');
const HOME = path.join(LOCAL, 'Eclipse');
const VENV = path.join(HOME, 'venv');
const DATA = path.join(HOME, 'data');

function root() {
  if (PACKAGED) return path.join(process.resourcesPath, 'backend');
  return path.resolve(__dirname, '..');
}
function venvPython() {
  return process.platform === 'win32' ? path.join(VENV, 'Scripts', 'python.exe') : path.join(VENV, 'bin', 'python');
}
function pythonExe() {
  if (PACKAGED) return venvPython();
  const r = root();
  for (const c of [path.join(r, '.venv', 'Scripts', 'python.exe'), path.join(r, '.venv', 'bin', 'python')]) if (fs.existsSync(c)) return c;
  return 'python';
}
function dataDir() { return PACKAGED ? DATA : root(); }
// extra env for every backend spawn (merge over process.env)
function env() {
  if (!PACKAGED) return {};
  try { fs.mkdirSync(DATA, { recursive: true }); } catch (e) {}
  return { ECLIPSE_DATA_DIR: DATA, PYTHONDONTWRITEBYTECODE: '1' };
}

// ---- first-run setup (packaged only) ----
const MARK = path.join(HOME, 'venv-ready.json');
function requirementsFile() { return path.join(root(), 'requirements-eclipse.txt'); }
function isReady() {
  if (!PACKAGED) return true;
  if (!fs.existsSync(venvPython())) return false;
  try { const m = JSON.parse(fs.readFileSync(MARK, 'utf8')); return m.requirements === fs.readFileSync(requirementsFile(), 'utf8'); } catch (e) { return false; }
}
// find a base interpreter able to create the venv: py -3.12, py -3, python
function findBasePython() {
  const cands = [['py', ['-3.12']], ['py', ['-3']], ['python', []], ['python3', []]];
  for (const [cmd, pre] of cands) {
    try {
      const out = execFileSync(cmd, [...pre, '-c', 'import sys;print("%d.%d"%sys.version_info[:2])'], { windowsHide: true, timeout: 15000, encoding: 'utf8' }).trim();
      const [maj, min] = out.split('.').map(Number);
      if (maj === 3 && min >= 10) return { cmd, pre, version: out };
    } catch (e) {}
  }
  return null;
}
function run(cmd, args, log) {
  return new Promise((resolve, reject) => {
    log('$ ' + [cmd, ...args].join(' '), 'sys');
    const ch = spawn(cmd, args, { windowsHide: true, env: Object.assign({}, process.env, { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', PIP_DISABLE_PIP_VERSION_CHECK: '1' }) });
    const pipe = (lvl) => { let b = ''; return (d) => { b += d; const p = b.split(/\r?\n|\r/); b = p.pop(); p.forEach((l) => l.trim() && log(l, lvl)); }; };
    ch.stdout.setEncoding('utf8'); ch.stdout.on('data', pipe('out'));
    ch.stderr.setEncoding('utf8'); ch.stderr.on('data', pipe('err'));
    ch.on('error', reject);
    ch.on('close', (code) => code === 0 ? resolve() : reject(new Error(cmd + ' exited with code ' + code)));
  });
}
let setupPromise = null;
// ensureReady({ send, dialogParent }) -> Promise<boolean>. Streams progress as studio-event log lines + 'backend-setup' events.
function ensureReady(opts) {
  if (isReady()) return Promise.resolve(true);
  if (setupPromise) return setupPromise;
  opts = opts || {};
  const send = opts.send || (() => {});
  const log = (message, level) => {
    const line = '[eclipse setup] ' + message;
    console.log(line);
    send('studio-event', { type: 'log', level: level || 'sys', message: line, slot: 'main' });
  };
  setupPromise = (async () => {
    send('backend-setup', { phase: 'start' });
    log('First run: preparing the Python environment in ' + VENV + ' (one time, a few minutes)');
    if (!fs.existsSync(venvPython())) {
      const base = findBasePython();
      if (!base) {
        log('No Python 3.10+ found (tried py -3.12, py -3, python).', 'err');
        send('backend-setup', { phase: 'error', error: 'python-not-found' });
        await noPythonDialog(opts.dialogParent);
        return false;
      }
      log('Using Python ' + base.version + ' (' + [base.cmd, ...base.pre].join(' ') + ')');
      fs.mkdirSync(HOME, { recursive: true });
      await run(base.cmd, [...base.pre, '-m', 'venv', VENV], log);
    }
    const py = venvPython();
    await run(py, ['-m', 'pip', 'install', '--upgrade', 'pip'], log).catch((e) => log('pip upgrade skipped: ' + e.message, 'err'));
    await run(py, ['-m', 'pip', 'install', '--progress-bar', 'off', '-r', requirementsFile()], log);
    fs.mkdirSync(DATA, { recursive: true });
    fs.writeFileSync(MARK, JSON.stringify({ requirements: fs.readFileSync(requirementsFile(), 'utf8'), created: new Date().toISOString() }));
    log('Python environment ready.');
    send('backend-setup', { phase: 'done' });
    return true;
  })().catch((e) => {
    log('Setup failed: ' + (e && e.message || e), 'err');
    send('backend-setup', { phase: 'error', error: String(e && e.message || e) });
    setupPromise = null;
    return false;
  });
  return setupPromise;
}
async function noPythonDialog(parent) {
  try {
    const { dialog, shell } = require('electron');
    const r = await dialog.showMessageBox(parent || null, {
      type: 'warning', title: 'Eclipse needs Python',
      message: 'Python 3.12 was not found on this PC.',
      detail: 'Eclipse runs its simulations with Python. Install Python 3.12 (64-bit) from python.org - tick "Add python.exe to PATH" ' +
        'or keep the "py launcher" option - then restart Eclipse. The first start after that sets up the environment automatically.',
      buttons: ['Open python.org', 'Close'], defaultId: 0, cancelId: 1,
    });
    if (r.response === 0) shell.openExternal('https://www.python.org/downloads/windows/');
  } catch (e) {}
}

module.exports = { PACKAGED, root, pythonExe, dataDir, env, ensureReady, isReady, HOME, VENV, DATA };

// CLI: `electron app/backend.js --setup` is not needed; for manual testing: node app/backend.js --info
if (require.main === module) console.log(JSON.stringify({ PACKAGED, root: root(), python: pythonExe(), data: dataDir(), ready: isReady() }, null, 2));
