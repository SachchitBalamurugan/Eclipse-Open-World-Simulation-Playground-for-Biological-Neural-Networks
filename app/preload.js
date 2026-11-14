const { contextBridge, ipcRenderer } = require('electron');
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
});
