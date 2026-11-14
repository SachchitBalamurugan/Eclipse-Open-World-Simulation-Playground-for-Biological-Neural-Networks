// npm run dist:win -> electron-builder --win portable zip (unsigned) into <repo>/dist, then publish the zip
// to landing/downloads/ with a manifest.json. Uses app/assets/icon.ico when present.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');

const APP = path.resolve(__dirname, '..');
const REPO = path.resolve(APP, '..');
const pkg = require(path.join(APP, 'package.json'));
const version = pkg.version;

const args = ['electron-builder', '--win', 'portable', 'zip', '--x64', '--publish', 'never'];
const icon = path.join(APP, 'assets', 'icon.ico');
if (fs.existsSync(icon)) args.push('-c.win.icon=assets/icon.ico');
else console.warn('[dist:win] app/assets/icon.ico not found - building with the default Electron icon');

const env = Object.assign({}, process.env, { CSC_IDENTITY_AUTO_DISCOVERY: 'false' });
execFileSync(process.platform === 'win32' ? 'npx.cmd' : 'npx', args, { cwd: APP, stdio: 'inherit', env, shell: process.platform === 'win32' });

const zipName = `Eclipse-${version}-win-x64.zip`;
const zip = path.join(REPO, 'dist', zipName);
if (!fs.existsSync(zip)) throw new Error('expected build output missing: ' + zip);
const outDir = path.join(REPO, 'landing', 'downloads');
fs.mkdirSync(outDir, { recursive: true });
const dst = path.join(outDir, zipName);
fs.copyFileSync(zip, dst);
const buf = fs.readFileSync(dst);
const manifest = {
  version,
  file: zipName,
  bytes: buf.length,
  sha256: crypto.createHash('sha256').update(buf).digest('hex'),
  built: new Date().toISOString(),
  requirements:
    'Windows 10/11 x64. Python 3.12 (64-bit, from python.org; 3.10+ works) must be installed and reachable as `py` or `python`. ' +
    'Unzip and run Eclipse.exe (unsigned - SmartScreen may ask you to confirm). On first run Eclipse creates a private Python ' +
    'environment in %LOCALAPPDATA%\\Eclipse\\venv and pip-installs numpy, mujoco, gymnasium and cl-sdk (internet required, a few ' +
    'hundred MB, a few minutes; progress is shown in the Studio log). Your runs and imported robots are stored in ' +
    '%LOCALAPPDATA%\\Eclipse\\data. If Python is missing, Eclipse shows a dialog linking to python.org.',
};
fs.writeFileSync(path.join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log('[dist:win] ' + dst + ' ' + manifest.bytes + ' bytes sha256 ' + manifest.sha256);
