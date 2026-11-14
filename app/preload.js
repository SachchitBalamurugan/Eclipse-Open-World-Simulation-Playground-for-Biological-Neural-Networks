const { contextBridge, ipcRenderer, webUtils } = require('electron');
const call = async (chan, ...args) => {
  const r = await ipcRenderer.invoke(chan, ...args);
  if (!r.ok) throw new Error(r.error);
  return r.data;
};
contextBridge.exposeInMainWorld('lab', {
  listRuns: () => call('list-runs'),
  loadRun: (name) => call('load-run', name),
  benchmarks: () => call('benchmarks'),
  conditions: () => call('conditions'),
  startRun: (opts) => call('start-run', opts),
  listLogs: () => call('list-logs'),
  tailLog: (name, pos) => call('tail-log', name, pos),
  selftestReport: (r) => call('selftest-report', r),
  fleetGet: () => call('fleet-get'),
  fleetPost: (p, body) => call('fleet-post', p, body),
  fleetPopout: (ids) => call('fleet-popout', ids),
  readAsset: (rel) => call('read-asset', rel),
  listReplays: () => call('list-replays'),
  loadReplay: (kind, run, file) => call('load-replay', kind, run, file),
  onReplaysChanged: (fn) => ipcRenderer.on('replays-changed', () => fn()),
  onRunLog: (fn) => ipcRenderer.on('run-log', (_e, p) => fn(p)),
  onRunDone: (fn) => ipcRenderer.on('run-done', (_e, p) => fn(p)),
  // Eclipse Studio (docs/STUDIO_CONTRACT.md)
  studioList: () => call('studio-list'),
  studioPickFile: () => call('studio-pick-file'),
  studioImport: (file, opts) => call('studio-import', file, opts),
  studioScene: (name) => call('studio-scene', name),
  studioTrain: (cfg) => call('studio-train', cfg),
  studioStop: (slot) => call('studio-stop', slot),
  studioRuns: () => call('studio-runs'),
  studioFleetStatus: () => call('studio-fleet-status'),
  transferTargets: () => call('transfer-targets'),
  transferExport: (runDir, target) => call('transfer-export', runDir, target),
  transferRun: (bundle, episodes) => call('transfer-run', bundle, episodes),
  transferReveal: (p) => call('transfer-reveal', p),
  studioReplay: (id) => call('studio-replay', id),
  studioDownload: (id) => call('studio-download', id),
  studioPack: (id) => call('studio-pack', id),
  studioDemoTeardown: () => call('studio-demo-teardown'),
  demoQuit: () => call('demo-quit'),
  onStudioEvent: (fn) => ipcRenderer.on('studio-event', (_e, p) => fn(p)),
  pathForFile: (file) => { try { return webUtils.getPathForFile(file) || ''; } catch (e) { return ''; } },
});
