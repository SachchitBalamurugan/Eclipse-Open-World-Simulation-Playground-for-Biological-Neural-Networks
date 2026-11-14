// BNN Robot Studio tab: import a robot, configure a simulated BNN controller, pick a platform, train, evaluate.
// Talks to the backend only through window.lab.studio* (main.js -> `python -m bnn_platform studio ...`, docs/STUDIO_CONTRACT.md).
// All biology is the simulated Izhikevich culture: no living neurons.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { gsap } from 'gsap';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v, d = 1) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '-');
const STEPS = ['Import robot', 'Configure BNN', 'Choose platform', 'Train', 'Evaluate', 'Deploy'];
const ACC = 0x76b900;

export function initStudio(root, lab, opts = {}) {
  root.innerHTML = `<div class="st">
  <div class="st-bar">
    <ol class="st-steps">${STEPS.map((s, i) => `<li data-step="${i + 1}"><b>${i + 1}</b><span>${s}</span></li>`).join('')}<div class="st-prog"><i></i></div></ol>
    <div class="st-pill" data-state="idle"><i></i><span>Idle</span></div>
  </div>
  <div class="st-main">
    <aside class="st-left st-panel">
      <div class="st-ph"><span>Robot library</span><button class="st-btn st-import" title="Import MJCF / URDF / zip">+ Import</button></div>
      <div class="st-filter"><input placeholder="Filter robots" spellcheck="false"></div>
      <div class="st-imports"></div>
      <div class="st-lib"></div>
      <div class="st-drophint"><b>Drop a robot file</b><span>.xml (MJCF) · .urdf · .zip with meshes</span></div>
    </aside>
    <section class="st-vp">
      <canvas></canvas>
      <div class="st-vtools"><button data-v="fit" title="Frame robot">Frame</button><button data-v="grid" class="on">Grid</button><button data-v="trail" class="on">Trail</button><button data-v="box" class="on">Target zone</button></div>
      <div class="st-vinfo"></div>
      <div class="st-vhud mono"></div>
      <div class="st-vmsg"><div class="st-spin"></div><b></b><span></span></div>
      <div class="st-banner"><span>Training started</span><em></em></div>
      <div class="st-summary"></div>
      <div class="st-vfoot mono">Z-up · metres · drag to orbit · wheel to zoom</div>
    </section>
    <aside class="st-right st-panel">
      <div class="st-ph"><span>Properties</span><em class="st-robotname"></em></div>
      <section class="st-grp" data-sec="robot"><h3>Robot</h3><div class="st-robotprops"></div></section>
      <section class="st-grp" data-sec="task"><h3>Task</h3>
        <div class="pg"><label>Task</label><select class="f-task"><option value="reach">Reach target (end-effector)</option></select></div>
      </section>
      <section class="st-grp" data-sec="bnn"><h3>BNN controller</h3>
        <div class="pg"><label>Neurons</label><input class="f-neurons" type="number" min="64" max="4096" step="64" value="1024"></div>
        <div class="pg"><label>Electrodes</label><input class="f-electrodes" type="number" value="64" readonly title="fixed 8x8 MEA"></div>
        <div class="pg"><label>Learning rule</label><div class="seg f-rule"><button data-v="teach" class="on">Teach</button><button data-v="frozen">Frozen</button></div></div>
        <div class="pg"><label>Learning rate</label><input class="f-lr" type="number" min="0.001" max="1" step="0.005" value="0.02"></div>
        <div class="pg"><label>Seed</label><input class="f-seed" type="number" min="0" max="1000000" step="1" value="0"></div>
        <p class="st-note"><i></i>Simulated Izhikevich culture on a virtual MEA - no living neurons.</p>
      </section>
      <section class="st-grp" data-sec="platform"><h3>Platform</h3>
        <div class="st-cards">
          <button class="st-card on" data-p="local"><b>Local CPU</b><span>Runs on this machine. Live viewport + spikes.</span></button>
          <button class="st-card" data-p="fleet"><b>Freestyle VM fleet</b><span>One seed per VM. Results stream back.</span></button>
        </div>
        <div class="st-fleetopts">
          <div class="pg"><label>VM count</label><div class="st-range"><input class="f-vms" type="range" min="1" max="16" value="4"><em class="mono">4</em></div></div>
          <div class="pg"><label>Dry run</label><label class="st-check"><input class="f-dry" type="checkbox" checked><span>Plan only, create no VMs</span></label></div>
        </div>
      </section>
      <section class="st-grp" data-sec="train"><h3>Training</h3>
        <div class="pg"><label>Episodes</label><input class="f-episodes" type="number" min="1" max="1000" step="1" value="20"></div>
        <div class="st-progress"><div><i></i></div><span class="mono">-</span></div>
        <div class="st-actions"><button class="st-train">Train</button><button class="st-stop" disabled>Stop</button></div>
        <div class="st-trainerr"></div>
      </section>
      <section class="st-grp" data-sec="deploy"><h3>Deploy to living culture <i class="tag">sim-to-real</i></h3>
        <p class="st-note warn"><i></i>Simulated weights don't transfer - the bundle carries the stimulation/decoding protocol, a calibration procedure and safety limits for the living culture.</p>
        <div class="pg"><label>Source run</label><select class="f-drun"></select></div>
        <div class="st-sub2">Target hardware</div>
        <div class="st-targets"><div class="st-empty">Loading targets...</div></div>
        <div class="st-actions"><button class="st-btn2 st-export">Export bundle</button><button class="st-btn2 st-dryrun" disabled title="export a bundle first">Dry-run closed loop (mock MEA)</button></div>
        <div class="st-bundle"></div>
      </section>
    </aside>
    <div class="st-dock">
      <div class="st-pane st-pchart"><div class="st-ph"><span>Error · mm</span><span class="st-legend"><i style="background:#76b900"></i>final<i style="background:#8a8a8a"></i>median</span></div><canvas class="c-curve"></canvas></div>
      <div class="st-pane st-pheat"><div class="st-ph"><span>Electrode spikes</span><em class="mono st-spk"></em></div><canvas class="c-heat"></canvas></div>
      <div class="st-pane st-pcon"><div class="st-ph"><span>Console</span><button class="st-btn st-clear">Clear</button></div><div class="st-con mono"></div></div>
      <div class="st-pane st-pruns"><div class="st-ph"><span class="st-tabs"><button data-t="runs" class="on">Run history</button><button data-t="jobs">Fleet jobs</button></span><button class="st-btn st-refresh" title="Refresh">&#x21bb;</button></div>
        <div class="st-runs"></div><div class="st-jobs" hidden></div></div>
    </div>
  </div>
  <div class="st-drop"><div><b>Drop to import robot</b><span>MJCF .xml · URDF · .zip (folder with meshes)</span></div></div>
</div>`;
  const q = (s) => root.querySelector(s);
  const qa = (s) => [...root.querySelectorAll(s)];

  const st = {
    active: false, robots: [], listError: null, sel: null, scene: null, sceneError: null, sceneLoading: false,
    step: 1, running: false, done: false, runId: null, episodes: [], nEp: 0, events: 0, frames: 0, lastFrame: null,
    spikes: null, platform: 'local', rule: 'teach', imports: [], jobs: [], runs: [], runsError: null, lastError: null, entered: false,
  };

  // ---------- motion (GSAP, reduced-motion aware) ----------
  const motion = { reduce: window.matchMedia('(prefers-reduced-motion: reduce)').matches };
  const mm = gsap.matchMedia();
  mm.add({ full: '(prefers-reduced-motion: no-preference)', reduce: '(prefers-reduced-motion: reduce)' }, (ctx) => { motion.reduce = !!ctx.conditions.reduce; });
  const D = (s) => (motion.reduce ? 0 : s);
  function enter() {
    if (st.entered) return; st.entered = true;
    if (motion.reduce) return;
    const tl = gsap.timeline({ defaults: { duration: 0.45, ease: 'power2.out', clearProps: 'transform,opacity' } });
    tl.from(qa('.st-steps li'), { y: -6, opacity: 0, stagger: 0.05 }, 0)
      .from(q('.st-left'), { x: -14, opacity: 0 }, 0.05)
      .from(q('.st-right'), { x: 14, opacity: 0 }, 0.1)
      .from(qa('.st-dock .st-pane'), { y: 10, opacity: 0, stagger: 0.05 }, 0.15)
      .from(qa('.st-right .st-grp'), { opacity: 0, y: 6, stagger: 0.04, duration: 0.3 }, 0.2);
  }
  function flash(el) {
    if (!el) return;
    gsap.fromTo(el, { backgroundColor: 'rgba(118,185,0,0.16)' }, { backgroundColor: 'rgba(118,185,0,0)', duration: D(0.9) || 0.01, ease: 'power1.out', clearProps: 'backgroundColor' });
  }

  // ---------- console ----------
  const con = q('.st-con');
  function log(msg, level = 'out') {
    const d = document.createElement('div'); d.className = 'l-' + level;
    const t = new Date(); d.innerHTML = `<time>${t.toTimeString().slice(0, 8)}</time>${esc(msg)}`;
    const stick = con.scrollTop + con.clientHeight >= con.scrollHeight - 8;
    con.appendChild(d); while (con.childNodes.length > 600) con.removeChild(con.firstChild);
    if (stick) con.scrollTop = con.scrollHeight;
  }
  q('.st-clear').onclick = () => { con.innerHTML = ''; };

  // ---------- stepper ----------
  function setStep(n, why) {
    n = Math.max(1, Math.min(STEPS.length, n));
    const prev = st.step; st.step = n;
    qa('.st-steps li').forEach((li) => { const k = Number(li.dataset.step); li.classList.toggle('done', k < n); li.classList.toggle('cur', k === n); });
    gsap.to(q('.st-prog i'), { scaleX: (n - 1) / (STEPS.length - 1), duration: D(0.5), ease: 'power2.inOut' });
    if (n !== prev && !motion.reduce) gsap.fromTo(q(`.st-steps li[data-step="${n}"] b`), { scale: 0.7 }, { scale: 1, duration: 0.4, ease: 'back.out(2.2)', clearProps: 'transform' });
  }
  const SEC_FOR_STEP = { 1: null, 2: 'bnn', 3: 'platform', 4: 'train', 5: null, 6: 'deploy' };
  qa('.st-steps li').forEach((li) => li.onclick = () => {
    const n = Number(li.dataset.step);
    if (n === 1) { flash(q('.st-left')); return; }
    if (!st.sel) { setStatus('idle', 'Select or import a robot first'); flash(q('.st-left')); return; }
    if (n === 5) { flash(q('.st-dock')); if (!st.running) setStep(5); return; }
    if (!st.running) setStep(n);
    const sec = q(`.st-grp[data-sec="${SEC_FOR_STEP[n]}"]`);
    if (sec) { sec.scrollIntoView({ block: 'nearest', behavior: motion.reduce ? 'auto' : 'smooth' }); flash(sec); }
  });
  q('.st-grp[data-sec="bnn"]').addEventListener('input', () => { if (st.sel && !st.running && st.step < 2) setStep(2); });
  function setStatus(state, text) {
    const p = q('.st-pill'); const changed = p.dataset.state !== state;
    p.dataset.state = state; p.querySelector('span').textContent = text;
    if (changed && !motion.reduce) gsap.fromTo(p, { scale: 0.94 }, { scale: 1, duration: 0.35, ease: 'back.out(2)', clearProps: 'transform' });
  }

  // ---------- form ----------
  const segBtns = qa('.f-rule button');
  segBtns.forEach((b) => b.onclick = () => { st.rule = b.dataset.v; segBtns.forEach((x) => x.classList.toggle('on', x === b)); if (st.sel && !st.running) setStep(Math.max(st.step, 2)); });
  function setPlatform(p) {
    st.platform = p;
    qa('.st-card').forEach((c) => c.classList.toggle('on', c.dataset.p === p));
    const fo = q('.st-fleetopts'); const show = p === 'fleet';
    if (show === !fo.hidden && fo.dataset.init) return;
    fo.dataset.init = '1';
    if (show) { fo.hidden = false; if (!motion.reduce) gsap.from(fo, { autoAlpha: 0, y: -6, duration: 0.3, ease: 'power2.out', clearProps: 'all' }); }
    else fo.hidden = true;
  }
  qa('.st-card').forEach((c) => c.onclick = () => { setPlatform(c.dataset.p); if (st.sel && !st.running) setStep(3); });
  setPlatform('local');
  q('.f-vms').oninput = () => { q('.f-vms + em').textContent = q('.f-vms').value; };
  function cfg() {
    return {
      robot: st.sel && st.sel.name, task: q('.f-task').value, platform: st.platform, rule: st.rule,
      neurons: Math.round(Number(q('.f-neurons').value)), electrodes: Number(q('.f-electrodes').value),
      lr: Number(q('.f-lr').value), seed: Math.round(Number(q('.f-seed').value)), episodes: Math.round(Number(q('.f-episodes').value)),
      vms: Number(q('.f-vms').value), dryRun: q('.f-dry').checked,
    };
  }
  function validate(c) {
    if (!c.robot) return 'select a robot first';
    if (!(c.neurons >= 64 && c.neurons <= 4096)) return 'neurons must be 64-4096';
    if (!(c.lr > 0 && c.lr <= 1)) return 'learning rate must be in (0, 1]';
    if (!(c.seed >= 0 && c.seed <= 1e6)) return 'seed must be 0-1,000,000';
    if (!(c.episodes >= 1 && c.episodes <= 1000)) return 'episodes must be 1-1000';
    return null;
  }

  // ---------- library ----------
  function robotCard(r) {
    const can = r.trainable !== false;
    return `<button class="st-robot${st.sel && st.sel.name === r.name ? ' sel' : ''}" data-name="${esc(r.name)}">
      <span class="st-rico">${esc((r.name || '?').slice(0, 2).toUpperCase())}</span>
      <span class="st-rtxt"><b>${esc(r.title || r.name)}</b><small class="mono">${esc(r.name)}</small>
        <em>${r.n_bodies ?? '-'} bodies · ${r.n_joints ?? '-'} joints · ${r.n_actuated ?? '-'} act.</em></span>
      <span class="st-rtags"><i class="tag ${r.kind === 'user' ? 'user' : ''}">${esc(r.kind || 'builtin')}</i>${can ? '' : '<i class="tag muted">view only</i>'}</span>
    </button>`;
  }
  function renderLib() {
    const f = q('.st-filter input').value.trim().toLowerCase();
    const list = st.robots.filter((r) => !f || (r.name + ' ' + (r.title || '')).toLowerCase().includes(f));
    const b = list.filter((r) => r.kind !== 'user'), u = list.filter((r) => r.kind === 'user');
    let html = '';
    if (st.listError) html += `<div class="st-err"><b>Robot library unavailable</b><span>${esc(st.listError)}</span><button class="st-btn st-retry">Retry</button></div>`;
    else if (!st.robots.length) html += `<div class="st-empty">Loading robots...</div>`;
    else {
      html += `<div class="st-sub">Built-in <em>${b.length}</em></div>` + (b.map(robotCard).join('') || '<div class="st-empty">no match</div>');
      html += `<div class="st-sub">Imported <em>${u.length}</em></div>` + (u.map(robotCard).join('') || '<div class="st-empty">No imported robots yet. Use + Import or drop a file.</div>');
    }
    q('.st-lib').innerHTML = html;
    qa('.st-robot').forEach((el) => el.onclick = () => selectRobot(el.dataset.name));
    const rt = q('.st-retry'); if (rt) rt.onclick = () => refreshList();
  }
  q('.st-filter input').oninput = renderLib;
  async function refreshList(selectName) {
    st.listError = null;
    try {
      const r = await lab.studioList();
      st.robots = Array.isArray(r && r.robots) ? r.robots : [];
      renderLib();
      if (!motion.reduce) gsap.from(qa('.st-robot'), { opacity: 0, x: -8, stagger: 0.03, duration: 0.3, ease: 'power2.out', clearProps: 'transform,opacity' });
      const want = selectName || (st.sel && st.sel.name) || (st.robots.find((x) => x.trainable !== false) || st.robots[0] || {}).name;
      if (want && st.robots.some((x) => x.name === want)) await selectRobot(want);
    } catch (e) {
      st.listError = e.message; st.robots = []; renderLib();
      log('studio list: ' + e.message, 'err');
      vmsg('Backend unavailable', e.message, false);
    }
  }

  // ---------- import ----------
  function importRow(file) {
    const el = document.createElement('div'); el.className = 'st-imp';
    el.innerHTML = `<div class="st-spin sm"></div><span><b>${esc(file.split(/[\\/]/).pop())}</b><em>importing...</em></span><button class="st-x" title="dismiss">×</button>`;
    el.querySelector('.st-x').onclick = () => el.remove();
    q('.st-imports').prepend(el);
    if (!motion.reduce) gsap.from(el, { height: 0, opacity: 0, duration: 0.25, ease: 'power2.out', clearProps: 'all' });
    return el;
  }
  async function importFile(file, iopts) {
    if (!file) return;
    const row = importRow(file); setStep(1);
    log('import ' + file, 'sys');
    try {
      const r = await lab.studioImport(file, iopts || {});
      const rob = r && r.robot;
      row.className = 'st-imp ok'; row.querySelector('.st-spin').remove();
      row.querySelector('em').textContent = 'imported as ' + (rob ? rob.name : '?') + (rob && rob.ee ? ' · EE ' + rob.ee : '');
      log('imported ' + (rob ? rob.name + ' (' + rob.n_bodies + ' bodies, ' + rob.n_actuated + ' actuated, ee ' + rob.ee + ')' : ''), 'ok');
      setTimeout(() => { if (row.isConnected) gsap.to(row, { opacity: 0, height: 0, duration: D(0.3), onComplete: () => row.remove() }); }, 6000);
      await refreshList(rob && rob.name);
    } catch (e) {
      row.className = 'st-imp bad'; const sp = row.querySelector('.st-spin'); if (sp) sp.remove();
      row.querySelector('em').textContent = e.message;
      log('import failed: ' + e.message, 'err');
      if (!motion.reduce) gsap.fromTo(row, { x: -4 }, { x: 0, duration: 0.4, ease: 'elastic.out(1, 0.3)', clearProps: 'transform' });
    }
  }
  q('.st-import').onclick = async () => {
    try { const f = await lab.studioPickFile(); if (f) await importFile(f); } catch (e) { log('file dialog: ' + e.message, 'err'); }
  };
  // drag-and-drop anywhere over the window while Studio is visible
  const drop = q('.st-drop'); let dragDepth = 0;
  const hasFiles = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes('Files');
  const visible = () => !root.hidden && root.offsetParent !== null;
  const showDrop = (on) => { if (drop.classList.contains('on') === on) return; drop.classList.toggle('on', on); gsap.to(drop, { autoAlpha: on ? 1 : 0, duration: D(0.18) }); };
  gsap.set(drop, { autoAlpha: 0 });
  document.addEventListener('dragenter', (e) => { if (!hasFiles(e)) return; e.preventDefault(); if (!visible()) return; dragDepth++; showDrop(true); });
  document.addEventListener('dragover', (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = visible() ? 'copy' : 'none'; });
  document.addEventListener('dragleave', (e) => { if (!hasFiles(e)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) showDrop(false); });
  document.addEventListener('drop', async (e) => {
    if (!hasFiles(e)) return; e.preventDefault(); dragDepth = 0; showDrop(false);
    if (!visible()) return;
    const files = [...e.dataTransfer.files];
    if (!files.length) return;
    for (const f of files) {
      const p = lab.pathForFile(f);
      if (!p) { const r = importRow(f.name); r.className = 'st-imp bad'; r.querySelector('.st-spin').remove(); r.querySelector('em').textContent = 'could not resolve a local path for this item'; continue; }
      if (!/\.(xml|urdf|zip)$/i.test(p)) { const r = importRow(p); r.className = 'st-imp bad'; r.querySelector('.st-spin').remove(); r.querySelector('em').textContent = 'unsupported file type: use .xml (MJCF), .urdf or .zip'; continue; }
      await importFile(p);
    }
  });

  // ---------- viewport ----------
  const canvas = q('.st-vp canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1c1e20);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
  camera.position.set(1.6, 1.2, 1.9);
  const controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.target.set(0, 0.3, 0);
  scene.add(new THREE.HemisphereLight(0xe8edf2, 0x202326, 1.25));
  const key = new THREE.DirectionalLight(0xffffff, 1.7); key.position.set(3, 6, 4); scene.add(key);
  const rim = new THREE.DirectionalLight(0xa8c8ff, 0.5); rim.position.set(-4, 3, -3); scene.add(rim);
  const gridG = new THREE.Group(); scene.add(gridG);
  function buildGrid(size) {
    gridG.clear();
    const major = new THREE.GridHelper(size, 10, 0x4a4f55, 0x34383d); major.material.transparent = true; major.material.opacity = 0.9;
    const minor = new THREE.GridHelper(size, 50, 0x2a2d31, 0x2a2d31); minor.position.y = -0.0005;
    const ax = new THREE.Group();
    const line = (a, b, c) => { const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]); return new THREE.Line(g, new THREE.LineBasicMaterial({ color: c })); };
    ax.add(line([-size / 2, 0.001, 0], [size / 2, 0.001, 0], 0x7a3a3a), line([0, 0.001, -size / 2], [0, 0.001, size / 2], 0x3a6a3a));
    gridG.add(minor, major, ax);
  }
  buildGrid(4);
  const mj = new THREE.Group(); mj.rotation.x = -Math.PI / 2; scene.add(mj); // MuJoCo z-up -> three y-up
  const robotG = new THREE.Group(); mj.add(robotG);
  let bodyGroups = [];
  const target = new THREE.Group();
  const tBall = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial({ color: ACC }));
  const tRing = new THREE.Mesh(new THREE.RingGeometry(1.6, 1.9, 40), new THREE.MeshBasicMaterial({ color: ACC, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }));
  target.add(tBall, tRing); target.visible = false; mj.add(target);
  const tipBall = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  tipBall.visible = false; mj.add(tipBall);
  const TRAIL_N = 4000, trailPos = new Float32Array(TRAIL_N * 3); let trailN = 0;
  const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trailGeo.setDrawRange(0, 0);
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffd23a, transparent: true, opacity: 0.85 })); trail.frustumCulled = false; mj.add(trail);
  let boxHelper = null, sceneScale = 0.5;
  const view = { grid: true, trail: true, box: true };
  qa('.st-vtools button').forEach((b) => b.onclick = () => {
    const v = b.dataset.v;
    if (v === 'fit') { fit(true); return; }
    view[v] = !view[v]; b.classList.toggle('on', view[v]);
    gridG.visible = view.grid; trail.visible = view.trail; if (boxHelper) boxHelper.visible = view.box;
  });

  const quat = (a) => new THREE.Quaternion(a[1], a[2], a[3], a[0]).normalize();
  function geomMesh(g) {
    const s = g.size || [], c = g.rgba || [0.6, 0.62, 0.65, 1];
    let geo = null;
    switch (g.type) {
      case 'box': geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
      case 'sphere': geo = new THREE.SphereGeometry(s[0], 24, 16); break;
      case 'capsule': geo = new THREE.CapsuleGeometry(s[0], 2 * (s[1] || 0), 8, 18); geo.rotateX(Math.PI / 2); break;
      case 'cylinder': geo = new THREE.CylinderGeometry(s[0], s[0], 2 * (s[1] || 0), 28); geo.rotateX(Math.PI / 2); break;
      case 'ellipsoid': geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s[0], s[1], s[2]); break;
      case 'plane': { const w = s[0] > 0 ? 2 * s[0] : 6, h = s[1] > 0 ? 2 * s[1] : 6; geo = new THREE.PlaneGeometry(w, h); break; }
      case 'mesh': {
        if (!g.mesh || !g.mesh.vertices || !g.mesh.faces) return null;
        geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.Float32BufferAttribute(g.mesh.vertices, 3));
        geo.setIndex(g.mesh.faces); geo.computeVertexNormals(); break;
      }
      default: return null;
    }
    const plane = g.type === 'plane';
    const col = new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
    const mat = plane ? new THREE.MeshStandardMaterial({ color: 0x232629, roughness: 0.95, transparent: true, opacity: 0.55, depthWrite: false })
      : new THREE.MeshStandardMaterial({ color: col, roughness: 0.45, metalness: 0.15, transparent: c[3] < 1, opacity: c[3] ?? 1, side: g.type === 'mesh' ? THREE.DoubleSide : THREE.FrontSide });
    const m = new THREE.Mesh(geo, mat);
    if (g.pos) m.position.set(...g.pos);
    if (g.quat) m.quaternion.copy(quat(g.quat));
    m.userData.plane = plane;
    if (plane) m.renderOrder = -1;
    return m;
  }
  function disposeRobot() {
    robotG.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    robotG.clear(); bodyGroups = [];
    if (boxHelper) { mj.remove(boxHelper); boxHelper = null; }
  }
  function buildScene(sc) {
    disposeRobot();
    const bodies = sc.bodies || [];
    bodyGroups = bodies.map((b, i) => {
      const g = new THREE.Group(); g.name = b.name || 'body' + i;
      if (b.pos) g.position.set(...b.pos);
      if (b.quat) g.quaternion.copy(quat(b.quat));
      robotG.add(g); return g;
    });
    let n = 0;
    for (const gm of sc.geoms || []) { const m = geomMesh(gm); const parent = bodyGroups[gm.body] || robotG; if (m) { parent.add(m); n++; } }
    if (sc.target_box && sc.target_box.min && sc.target_box.max) {
      boxHelper = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...sc.target_box.min), new THREE.Vector3(...sc.target_box.max)), new THREE.Color(0x4f7a10));
      boxHelper.visible = view.box; mj.add(boxHelper);
    }
    resetTrail(); target.visible = false; tipBall.visible = false;
    fit(false);
    return n;
  }
  const _box = new THREE.Box3(), _tmp = new THREE.Box3();
  function fit(animate) {
    scene.updateMatrixWorld(true);
    _box.makeEmpty();
    robotG.traverse((o) => { if (o.isMesh && !o.userData.plane) { o.geometry.computeBoundingBox(); _tmp.copy(o.geometry.boundingBox).applyMatrix4(o.matrixWorld); _box.union(_tmp); } });
    if (_box.isEmpty()) _box.set(new THREE.Vector3(-0.5, 0, -0.5), new THREE.Vector3(0.5, 1, 0.5));
    const c = _box.getCenter(new THREE.Vector3()), r = Math.max(0.05, _box.getSize(new THREE.Vector3()).length() / 2);
    sceneScale = r;
    const tr = Math.max(0.004, r * 0.018); tBall.scale.setScalar(tr); tRing.scale.setScalar(tr); tipBall.scale.setScalar(tr * 0.6);
    const pos = c.clone().add(new THREE.Vector3(1, 0.62, 1.15).normalize().multiplyScalar(r * 2.7));
    camera.near = r / 200; camera.far = r * 400; camera.updateProjectionMatrix();
    const gs = Math.pow(10, Math.ceil(Math.log10(r * 6))); buildGrid(gs); gridG.visible = view.grid;
    controls.maxDistance = r * 30;
    if (animate && !motion.reduce) {
      gsap.to(camera.position, { x: pos.x, y: pos.y, z: pos.z, duration: 0.8, ease: 'power3.inOut' });
      gsap.to(controls.target, { x: c.x, y: c.y, z: c.z, duration: 0.8, ease: 'power3.inOut' });
    } else { camera.position.copy(pos); controls.target.copy(c); }
  }
  function resetTrail() { trailN = 0; trailGeo.setDrawRange(0, 0); }
  function pushTrail(p) {
    if (trailN >= TRAIL_N) { trailPos.copyWithin(0, 3); trailN = TRAIL_N - 1; }
    trailPos.set(p, trailN * 3); trailN++;
    trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, trailN);
  }
  function vmsg(title, sub, spin) {
    const m = q('.st-vmsg');
    if (!title) { m.classList.remove('on'); return; }
    m.querySelector('b').textContent = title; m.querySelector('span').textContent = sub || '';
    m.querySelector('.st-spin').style.display = spin ? '' : 'none';
    if (!m.classList.contains('on')) { m.classList.add('on'); if (!motion.reduce) gsap.from(m, { opacity: 0, y: 6, duration: 0.3, clearProps: 'all' }); }
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    drawCurve(); drawHeat();
  }
  new ResizeObserver(resize).observe(q('.st-vp'));
  new ResizeObserver(() => { drawCurve(); drawHeat(); }).observe(q('.st-dock'));
  let last = performance.now(), fps = 0, fN = 0, fT = 0;
  function frame(now) {
    if (!st.active) return;
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    controls.update();
    if (target.visible) tRing.rotation.z += dt * 0.8;
    renderer.render(scene, camera);
    fN++; fT += dt; if (fT > 0.5) { fps = fN / fT; fN = 0; fT = 0; hud(); }
    requestAnimationFrame(frame);
  }
  function hud() {
    const f = st.lastFrame;
    q('.st-vhud').textContent = `${fps.toFixed(0)} fps` + (f ? `  ·  ep ${f.episode}  step ${f.step}  ·  err ${fmt(f.err_mm)} mm` : '');
  }

  // ---------- robot selection ----------
  function renderRobotProps() {
    const r = st.sel;
    q('.st-robotname').textContent = r ? r.name : '';
    if (!r) { q('.st-robotprops').innerHTML = '<p class="st-empty">No robot selected.</p>'; return; }
    const rows = [['Title', esc(r.title || r.name)], ['Kind', esc(r.kind || 'builtin')], ['Bodies', r.n_bodies ?? '-'], ['Joints', r.n_joints ?? '-'],
      ['Actuated', r.n_actuated ?? '-'], ['End-effector', `<span class="mono">${esc(r.ee || '-')}</span>`], ['Tasks', esc((r.tasks || []).join(', ') || 'none')]];
    if (r.source) rows.push(['Source', `<span class="mono" title="${esc(r.source)}">${esc(r.source)}</span>`]);
    q('.st-robotprops').innerHTML = rows.map(([k, v]) => `<div class="pg ro"><label>${k}</label><div>${v}</div></div>`).join('') +
      (r.trainable === false ? `<p class="st-warn">Shown for inspection. This robot trains through its own pipeline, not Studio.</p>` : '');
    const ts = q('.f-task'); ts.innerHTML = (r.tasks && r.tasks.length ? r.tasks : ['reach']).map((t) => `<option value="${esc(t)}">${t === 'reach' ? 'Reach target (end-effector)' : esc(t)}</option>`).join('');
    syncButtons();
  }
  async function selectRobot(name) {
    const r = st.robots.find((x) => x.name === name); if (!r) return;
    if (st.running && (!st.sel || st.sel.name !== name)) { log('cannot switch robot while training', 'err'); return; }
    const same = st.sel && st.sel.name === name && st.scene;
    st.sel = r; renderLib(); renderRobotProps();
    if (!st.running && st.step < 2) setStep(2);
    if (same) return;
    st.scene = null; st.sceneError = null; st.sceneLoading = true;
    vmsg('Loading ' + (r.title || r.name), 'python -m bnn_platform studio scene --robot ' + r.name, true);
    q('.st-vinfo').innerHTML = '';
    const t0 = performance.now();
    try {
      const sc = await lab.studioScene(name);
      if (!st.sel || st.sel.name !== name) return;
      st.scene = sc; const n = buildScene(sc);
      st.sceneGeoms = n; st.sceneBodies = (sc.bodies || []).length;
      vmsg(null);
      q('.st-vinfo').innerHTML = `<b>${esc(r.title || r.name)}</b><span class="mono">${st.sceneBodies} bodies · ${n} geoms · EE ${esc(sc.ee || r.ee || '-')}</span>`;
      log(`scene ${name}: ${st.sceneBodies} bodies, ${n} geoms (${((performance.now() - t0) / 1000).toFixed(1)} s)`, 'sys');
      if (!motion.reduce) gsap.from(robotG.scale, { x: 0.96, y: 0.96, z: 0.96, duration: 0.5, ease: 'power2.out' });
    } catch (e) {
      if (!st.sel || st.sel.name !== name) return;
      st.sceneError = e.message; disposeRobot();
      vmsg('Could not load scene', e.message, false);
      log('scene ' + name + ': ' + e.message, 'err');
    } finally { st.sceneLoading = false; }
  }

  // ---------- training ----------
  function syncButtons() {
    const can = !!st.sel && st.sel.trainable !== false;
    q('.st-train').disabled = st.running || !can || (st.platform === 'fleet' && !!st.fleetRunning);
    q('.st-train').textContent = st.running ? 'Training...' : (st.platform === 'fleet' ? (q('.f-dry').checked ? 'Plan fleet run' : 'Launch on fleet') : 'Train');
    q('.st-stop').disabled = !st.running;
    qa('.st-right input, .st-right select, .st-right .seg button, .st-card, .st-export, .st-target').forEach((el) => { el.disabled = st.running; });
    q('.st-dryrun').disabled = st.running || !st.bundle;
  }
  qa('.st-card, .f-dry').forEach((el) => el.addEventListener('click', () => setTimeout(syncButtons)));
  function trainErr(msg) {
    const el = q('.st-trainerr'); el.textContent = msg || '';
    if (msg && !motion.reduce) gsap.fromTo(el, { x: -4 }, { x: 0, duration: 0.4, ease: 'elastic.out(1, 0.3)', clearProps: 'transform' });
  }
  function setProgress(k, n) {
    const p = n ? Math.min(1, k / n) : 0;
    gsap.to(q('.st-progress i'), { scaleX: p, duration: D(0.4), ease: 'power2.out' });
    q('.st-progress span').textContent = n ? `${k} / ${n} episodes` : '-';
  }
  function hideSummary() { const s = q('.st-summary'); s.classList.remove('on'); s.innerHTML = ''; }
  async function train() {
    const c = cfg(); const bad = validate(c);
    if (bad) { trainErr(bad); return; }
    trainErr('');
    if (c.platform === 'fleet') {
      if (st.fleetRunning) { trainErr('a fleet launch is already in progress'); return; }
      st.fleetRunning = true; st.jobs = []; renderJobs(); syncButtons();
      try { await lab.studioTrain(c); setStep(4); setStatus('run', 'Launching fleet'); startSequence(c); showTab('jobs'); }
      catch (e) { st.fleetRunning = false; syncButtons(); trainErr(e.message); setStatus('err', 'Error'); log('fleet: ' + e.message, 'err'); }
      return;
    }
    hideSummary();
    st.episodes = []; st.nEp = c.episodes; st.done = false; st.lastFrame = null; st.spikes = null; st.jobs = [];
    resetTrail(); drawCurve(); drawHeat(); setProgress(0, c.platform === 'local' ? c.episodes : 0);
    st.running = true; st.mode = 'train'; syncButtons();
    try {
      await lab.studioTrain(c);
      setStep(4); setStatus('run', c.platform === 'fleet' ? 'Launching fleet' : 'Training');
      startSequence(c);
    } catch (e) {
      st.running = false; syncButtons(); trainErr(e.message); setStatus('err', 'Error'); log('train: ' + e.message, 'err');
    }
  }
  function startSequence(c) {
    const b = q('.st-banner');
    b.querySelector('em').textContent = `${c.robot} · ${c.rule} · ${c.platform === 'fleet' ? c.vms + ' VMs' + (c.dryRun ? ' (dry run)' : '') : 'local CPU'} · ${c.episodes} ep`;
    if (motion.reduce) { gsap.set(b, { autoAlpha: 1 }); gsap.delayedCall(2.5, () => gsap.set(b, { autoAlpha: 0 })); return; }
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
    tl.fromTo(q('.st-train'), { scale: 0.96 }, { scale: 1, duration: 0.3, clearProps: 'transform' }, 0)
      .fromTo(b, { autoAlpha: 0, y: -12 }, { autoAlpha: 1, y: 0, duration: 0.4 }, 0.05)
      .fromTo(b.querySelector('span'), { letterSpacing: '0.4em' }, { letterSpacing: '0.16em', duration: 0.6 }, 0.05)
      .fromTo(qa('.st-dock .st-pane'), { boxShadow: 'inset 0 2px 0 rgba(118,185,0,0)' }, { boxShadow: 'inset 0 2px 0 rgba(118,185,0,1)', duration: 0.25, stagger: 0.07 }, 0.2)
      .to(qa('.st-dock .st-pane'), { boxShadow: 'inset 0 2px 0 rgba(118,185,0,0)', duration: 0.6, stagger: 0.07, clearProps: 'boxShadow' }, 0.65)
      .to(b, { autoAlpha: 0, y: -8, duration: 0.4, ease: 'power2.in' }, 2.4);
  }
  async function stop() {
    try { const r = await lab.studioStop(); log(r && r.stopped ? 'stop requested (process tree killed)' : 'nothing to stop', 'sys'); } catch (e) { log('stop: ' + e.message, 'err'); }
  }
  q('.st-train').onclick = train;
  q('.st-stop').onclick = stop;

  function applyFrame(f) {
    st.frames++; st.lastFrame = f;
    if (Array.isArray(f.bodies)) {
      for (let i = 1; i < f.bodies.length && i < bodyGroups.length; i++) {
        const b = f.bodies[i]; if (!b || b.length < 7) continue;
        bodyGroups[i].position.set(b[0], b[1], b[2]); bodyGroups[i].quaternion.set(b[4], b[5], b[6], b[3]).normalize();
      }
    }
    if (Array.isArray(f.target)) { target.position.set(...f.target); target.visible = true; }
    if (Array.isArray(f.tip)) {
      tipBall.position.set(...f.tip); tipBall.visible = true;
      if (st.trailEp !== f.episode) { st.trailEp = f.episode; resetTrail(); }
      pushTrail(f.tip);
    }
    if (Array.isArray(f.spikes)) {
      const n = f.spikes.length;
      if (!st.spikes || st.spikes.length !== n) st.spikes = new Float32Array(n);
      for (let i = 0; i < n; i++) st.spikes[i] = st.spikes[i] * 0.55 + (Number(f.spikes[i]) || 0);
      drawHeat();
    }
  }
  function onFleetEvent(o) {
    switch (o.type) {
      case 'fleet':
        st.jobs = o.jobs || []; renderJobs();
        log(`fleet${o.dry_run ? ' (dry run, no VMs created)' : ''}: ` + st.jobs.map((j) => j.slug + '=' + j.state).join(', '), 'sys'); break;
      case 'start': log('fleet run ' + o.run_id + ' launching', 'ok'); break;
      case 'log': log(o.message, o.level || 'out'); break;
      case 'done': log(o.dry_run ? 'fleet plan complete (dry run)' : 'fleet launch complete: ' + JSON.stringify(o.summary || {}).slice(0, 200), 'ok'); if (!st.running) setStatus('ok', o.dry_run ? 'Fleet planned' : 'Fleet running'); break;
      case 'error': log('fleet: ' + o.message, 'err'); break;
      case 'exit': st.fleetRunning = false; syncButtons(); break;
    }
  }
  function onEvent(o) {
    st.events++;
    if (o.slot === 'fleet') { onFleetEvent(o); return; }
    switch (o.type) {
      case 'start':
        if (st.mode !== 'deploy') st.runId = o.run_id;
        log(`run ${o.run_id} started (${o.biology === 'simulated' ? 'simulated culture' : 'biology: ' + o.biology})`, 'ok');
        break;
      case 'frame': applyFrame(o); break;
      case 'episode': {
        st.episodes.push(o); if (o.episodes) st.nEp = o.episodes;
        setProgress(st.episodes.length, st.nEp);
        setStatus('run', `Training · ep ${o.episode + (o.episode === 0 ? 1 : 0)}/${st.nEp}`);
        log(`episode ${o.episode}: final ${fmt(o.final_err_mm)} mm, median ${fmt(o.median_err_mm)} mm, ${o.success ? 'SUCCESS' : 'miss'}, success rate ${fmt((o.success_rate || 0) * 100, 0)}%`, o.success ? 'ok' : 'out');
        drawCurve(); break;
      }
      case 'log': log(o.message, o.level || 'out'); break;
      case 'calibration': {
        const { type, stage, slot, ...rest } = o;
        log(`calibration · ${stage}: ` + Object.entries(rest).filter(([, v]) => typeof v !== 'object').map(([k, v]) => k + '=' + v).join(', ') + (Array.isArray(rest.changes) ? `, ${rest.changes.length} electrode remap(s)` : ''), rest.accepted === false ? 'warn' : 'sys');
        setStatus('run', 'Calibrating · ' + stage); break;
      }
      case 'fleet':
        st.jobs = o.jobs || []; renderJobs(); showTab('jobs');
        log(`fleet${o.dry_run ? ' (dry run, no VMs created)' : ''}: ${st.jobs.length} job(s) ` + st.jobs.map((j) => j.slug + '=' + j.state).join(', '), 'sys');
        break;
      case 'done': {
        if (st.mode === 'replay') { setStatus('ok', 'Replay complete'); showSummary(o.summary || {}, o.run_dir, 'Replay · ' + (st.replayId || '')); refreshRuns(); break; }
        if (st.mode === 'deploy') {
          const s = o.summary || {}; setStatus('ok', 'Dry run complete'); setStep(6);
          log('closed-loop dry run (mock MEA) done: ' + JSON.stringify(s).slice(0, 240), 'ok');
          showSummary(s, o.run_dir || (st.bundle && st.bundle.bundle), 'Closed-loop dry run · mock MEA'); break;
        }
        st.done = true; setStep(5); setStatus('ok', 'Complete'); st.lastRunDir = o.run_dir || null;
        const s = o.summary || {};
        log(`done: success ${fmt((s.success_rate || 0) * 100, 0)}%, median final error ${fmt(s.median_final_err_mm)} mm over ${s.episodes ?? '-'} episodes -> ${o.run_dir || ''}`, 'ok');
        showSummary(s, o.run_dir); refreshRuns(); break;
      }
      case 'error': st.lastError = o.message; setStatus('err', o.message === 'stopped by user' ? 'Stopped' : 'Error'); trainErr(o.message === 'stopped by user' ? '' : o.message); log(o.message, 'err'); break;
      case 'exit': st.running = false; syncButtons(); if (st.mode === 'train' && !st.done && st.step === 4) setStep(3); break;
      default: log(JSON.stringify(o).slice(0, 300), 'out');
    }
  }
  lab.onStudioEvent(onEvent);
  function showSummary(s, dir, title) {
    const el = q('.st-summary');
    el.innerHTML = `<h4>${esc(title || 'Evaluation')}</h4><div class="st-kpis">
      <div><span>Success rate</span><b>${fmt((s.success_rate || 0) * 100, 0)}<small>%</small></b></div>
      <div><span>Median final error</span><b>${fmt(s.median_final_err_mm)}<small>mm</small></b></div>
      <div><span>Episodes</span><b>${s.episodes ?? '-'}</b></div></div>
      ${s.eval_before || s.eval_after ? `<div class="st-heldout"><span>Held-out error before → after</span><b>${fmt(s.eval_before && s.eval_before.median_final_err_mm)} → ${fmt(s.eval_after && s.eval_after.median_final_err_mm)}<small>mm</small></b><em>learning off · success ${fmt(((s.eval_before && s.eval_before.success_rate) || 0) * 100, 0)}% → ${fmt(((s.eval_after && s.eval_after.success_rate) || 0) * 100, 0)}%</em></div>` : ''}
      <p class="mono">${esc(dir || '')}</p><button class="st-x" title="close">×</button>`;
    el.querySelector('.st-x').onclick = hideSummary;
    el.classList.add('on');
    if (!motion.reduce) {
      gsap.timeline({ defaults: { ease: 'power3.out' } }).from(el, { autoAlpha: 0, y: 14, duration: 0.45, clearProps: 'all' })
        .from(el.querySelectorAll('.st-kpis > div'), { autoAlpha: 0, y: 8, stagger: 0.08, duration: 0.35, clearProps: 'all' }, 0.15);
    }
  }

  // ---------- charts ----------
  function sizeCanvas(c) {
    const r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(10, Math.round(r.width * dpr)), h = Math.max(10, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return { g: c.getContext('2d'), w, h, dpr };
  }
  function drawCurve() {
    const c = q('.c-curve'); if (!c.isConnected || !c.clientWidth) return;
    const { g, w, h, dpr } = sizeCanvas(c);
    g.clearRect(0, 0, w, h);
    const L = 40 * dpr, R = 10 * dpr, T = 10 * dpr, B = 20 * dpr;
    const eps = st.episodes, n = Math.max(st.nEp || 0, eps.length, 2);
    const vals = eps.flatMap((e) => [e.final_err_mm, e.median_err_mm]).filter((v) => Number.isFinite(v));
    const ymax = vals.length ? Math.max(10, Math.max(...vals) * 1.1) : 100;
    const X = (i) => L + (w - L - R) * (i / (n - 1)), Y = (v) => T + (h - T - B) * (1 - v / ymax);
    g.font = `${10 * dpr}px "JetBrains Mono", Consolas, monospace`; g.textBaseline = 'middle';
    for (let k = 0; k <= 4; k++) {
      const v = ymax * k / 4, y = Y(v);
      g.strokeStyle = k ? '#262626' : '#3a3a3a'; g.lineWidth = 1; g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
      g.fillStyle = '#7a7a7a'; g.textAlign = 'right'; g.fillText(v.toFixed(0), L - 6 * dpr, y);
    }
    g.textAlign = 'center'; g.textBaseline = 'top';
    const tick = Math.max(1, Math.ceil(n / 8));
    for (let i = 0; i < n; i += tick) { g.fillStyle = '#6a6a6a'; g.fillText(String(i + 1), X(i), h - B + 5 * dpr); }
    if (!eps.length) {
      g.fillStyle = '#5f5f5f'; g.textBaseline = 'middle'; g.font = `${11 * dpr}px Inter, "Segoe UI", sans-serif`;
      g.fillText(st.running ? 'waiting for first episode...' : 'Train a robot to see its learning curve', (L + w - R) / 2, (T + h - B) / 2); return;
    }
    const line = (key, col, lw) => {
      g.strokeStyle = col; g.lineWidth = lw * dpr; g.beginPath(); let first = true;
      eps.forEach((e, i) => { const v = e[key]; if (!Number.isFinite(v)) return; const x = X(i), y = Y(v); if (first) { g.moveTo(x, y); first = false; } else g.lineTo(x, y); });
      g.stroke();
    };
    // fill under final-error line
    g.beginPath(); eps.forEach((e, i) => { const v = Number.isFinite(e.final_err_mm) ? e.final_err_mm : 0; i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); });
    g.lineTo(X(eps.length - 1), Y(0)); g.lineTo(X(0), Y(0)); g.closePath();
    const grd = g.createLinearGradient(0, T, 0, h - B); grd.addColorStop(0, 'rgba(118,185,0,0.22)'); grd.addColorStop(1, 'rgba(118,185,0,0)'); g.fillStyle = grd; g.fill();
    line('median_err_mm', '#8a8a8a', 1.2); line('final_err_mm', '#76b900', 1.8);
    eps.forEach((e, i) => { if (!Number.isFinite(e.final_err_mm)) return; g.fillStyle = e.success ? '#76b900' : '#1a1a1a'; g.strokeStyle = '#76b900'; g.lineWidth = 1.2 * dpr; g.beginPath(); g.arc(X(i), Y(e.final_err_mm), 2.6 * dpr, 0, Math.PI * 2); g.fill(); g.stroke(); });
  }
  function heatColor(t) {
    t = Math.max(0, Math.min(1, t));
    if (t < 0.6) { const k = t / 0.6; return [Math.round(24 + (118 - 24) * k), Math.round(26 + (185 - 26) * k), Math.round(28 * (1 - k))]; }
    const k = (t - 0.6) / 0.4; return [Math.round(118 + 137 * k), Math.round(185 + 70 * k), Math.round(255 * k)];
  }
  function drawHeat() {
    const c = q('.c-heat'); if (!c.isConnected || !c.clientWidth) return;
    const { g, w, h, dpr } = sizeCanvas(c);
    g.clearRect(0, 0, w, h);
    const n = st.spikes ? st.spikes.length : 64, side = Math.ceil(Math.sqrt(n));
    const sz = Math.min(w - 16 * dpr, h - 16 * dpr), cell = sz / side, x0 = (w - sz) / 2, y0 = (h - sz) / 2, gap = Math.max(1, cell * 0.08);
    const mx = st.spikes ? Math.max(1, ...st.spikes) : 1;
    let tot = 0;
    for (let i = 0; i < side * side; i++) {
      const v = st.spikes && i < n ? st.spikes[i] : 0; tot += v;
      const [r, gg, b] = i < n ? heatColor(v / mx) : [16, 16, 16];
      g.fillStyle = `rgb(${r},${gg},${b})`;
      g.fillRect(x0 + (i % side) * cell + gap / 2, y0 + Math.floor(i / side) * cell + gap / 2, cell - gap, cell - gap);
    }
    q('.st-spk').textContent = st.spikes ? `${side}×${side} · Σ ${tot.toFixed(0)}` : `${side}×${side} MEA`;
  }

  // ---------- run history / jobs ----------
  function showTab(t) {
    qa('.st-tabs button').forEach((b) => b.classList.toggle('on', b.dataset.t === t));
    q('.st-runs').hidden = t !== 'runs'; q('.st-jobs').hidden = t !== 'jobs';
  }
  qa('.st-tabs button').forEach((b) => b.onclick = () => { showTab(b.dataset.t); if (b.dataset.t === 'jobs') refreshJobs(); });
  q('.st-refresh').onclick = () => (q('.st-jobs').hidden ? refreshRuns() : refreshJobs());
  async function refreshRuns() {
    try {
      const r = await lab.studioRuns(); st.runs = ((r && r.runs) || []).slice().sort((a, b) => String(b.created || '').localeCompare(String(a.created || ''))); st.runsError = null;
    } catch (e) { st.runsError = e.message; st.runs = []; }
    fillDeployRuns();
    const el = q('.st-runs');
    if (st.runsError) { el.innerHTML = `<div class="st-err"><b>Run history unavailable</b><span>${esc(st.runsError)}</span></div>`; return; }
    if (!st.runs.length) { el.innerHTML = '<div class="st-empty">No Studio runs yet.</div>'; return; }
    el.innerHTML = '<table><tr><th>run</th><th>robot</th><th>date</th><th>success</th><th>med err</th><th></th></tr>' +
      st.runs.map((r) => { const s = r.summary || {}; return `<tr data-run="${esc(r.run_id)}" class="${r.run_id === st.runId ? 'sel' : ''}" title="${esc(r.run_id + ' · ' + (r.robot || '') + ' · ' + (r.rule || '') + ' · ' + (r.platform || '') + ' · ' + (r.created || ''))}"><td>${esc(r.run_id)}</td><td>${esc(r.robot || '-')}</td><td>${esc(String(r.created || '').replace('T', ' ').slice(0, 16))}</td><td>${s.success_rate == null ? '-' : fmt(s.success_rate * 100, 0) + '%'}</td><td>${fmt(s.median_final_err_mm)}</td><td class="st-rbtns"><button class="st-btn st-rplay" title="Replay into viewport">&#9654;</button><button class="st-btn st-rdl" title="Download run (.zip)">&#11015;</button></td></tr>`; }).join('') + '</table>';
    qa('.st-runs .st-rplay').forEach((b) => b.onclick = (e) => { e.stopPropagation(); replayRun(b.closest('tr').dataset.run); });
    qa('.st-runs .st-rdl').forEach((b) => b.onclick = async (e) => {
      e.stopPropagation(); const id = b.closest('tr').dataset.run; b.disabled = true; log('packing ' + id, 'sys');
      try { const r = await lab.studioDownload(id); log(r.saved ? 'saved ' + r.saved : 'download cancelled (zip at ' + r.zip + ')', 'ok'); } catch (err) { log('download: ' + err.message, 'err'); }
      b.disabled = false;
    });
    qa('.st-runs tr[data-run]').forEach((tr) => tr.onclick = () => { st.runId = tr.dataset.run; qa('.st-runs tr').forEach((x) => x.classList.toggle('sel', x === tr)); fillDeployRuns('user_runs/' + tr.dataset.run); });
  }
  function renderJobs(err) {
    const el = q('.st-jobs');
    if (err) { el.innerHTML = `<div class="st-err"><b>Fleet status unavailable</b><span>${esc(err)}</span></div>`; return; }
    if (!st.jobs.length) { el.innerHTML = '<div class="st-empty">No fleet jobs. Choose the Freestyle VM fleet platform and train.</div>'; return; }
    el.innerHTML = '<table><tr><th>job</th><th>vm</th><th>state</th><th>last ep</th><th>err mm</th></tr>' + st.jobs.map((j) => `<tr><td>${esc(j.slug)}</td><td>${esc(j.vm || '-')}</td><td><i class="state ${esc(j.state)}">${esc(j.state)}</i></td><td>${j.last ? esc(j.last.episode) : '-'}</td><td>${j.last ? fmt(j.last.final_err_mm) : '-'}</td></tr>`).join('') + '</table>';
  }
  async function refreshJobs() {
    try { const r = await lab.studioFleetStatus(); if (r && Array.isArray(r.jobs) && r.jobs.length) st.jobs = r.jobs; renderJobs(); } catch (e) { renderJobs(st.jobs.length ? null : e.message); }
  }

  async function replayRun(id) {
    if (st.running) { log('a run is active - stop it first', 'err'); return; }
    const run = st.runs.find((r) => r.run_id === id);
    if (run && run.robot && st.robots.some((x) => x.name === run.robot) && (!st.sel || st.sel.name !== run.robot)) await selectRobot(run.robot);
    hideSummary(); st.episodes = []; st.nEp = (run && run.summary && run.summary.episodes) || 0; st.lastFrame = null; st.spikes = null; resetTrail(); drawCurve(); drawHeat(); setProgress(0, st.nEp);
    st.running = true; st.mode = 'replay'; st.replayId = id; syncButtons();
    try { await lab.studioReplay(id); setStatus('run', 'Replaying ' + id); log('replay ' + id, 'sys'); }
    catch (e) { st.running = false; syncButtons(); setStatus('err', 'Error'); log('replay: ' + e.message, 'err'); }
  }
  // ---------- deploy: sim-to-real export (python -m bnn_platform.transfer) ----------
  const SURG_PKL = 'surgical/results/culture_teach.pkl';
  function fillDeployRuns(want) {
    const sel = q('.f-drun'); const fresh = st.lastRunDir ? String(st.lastRunDir).replace(/\\/g, '/') : ''; st.lastRunDir = null;
    const cur = want || fresh || sel.value;
    const opts = st.runs.map((r) => ({ v: 'user_runs/' + r.run_id, t: r.run_id + (r.summary && r.summary.success_rate != null ? '  ·  ' + fmt(r.summary.success_rate * 100, 0) + '%' : '') }));
    opts.push({ v: SURG_PKL, t: 'Surgical demo culture (culture_teach.pkl)' });
    sel.innerHTML = opts.map((o) => `<option value="${esc(o.v)}">${esc(o.t)}</option>`).join('');
    const hit = opts.find((o) => cur && (o.v === cur || cur.endsWith(o.v)));
    if (hit) sel.value = hit.v;
  }
  st.target = 'generic'; st.targets = [];
  function renderTargets(err) {
    const el = q('.st-targets');
    if (err) { el.innerHTML = `<div class="st-err"><b>Targets unavailable</b><span>${esc(err)}</span><button class="st-btn st-tretry">Retry</button></div>`; q('.st-tretry').onclick = loadTargets; return; }
    if (!st.targets.length) { el.innerHTML = '<div class="st-empty">Loading targets...</div>'; return; }
    el.innerHTML = st.targets.map((t) => `<button class="st-target${t.id === st.target ? ' on' : ''}" data-id="${esc(t.id)}" title="${esc((t.vendor ? t.vendor + ' · ' : '') + (t.n_electrodes ? t.n_electrodes + ' electrodes · ' : '') + 'SDK: ' + (t.sdk || '-'))}"><i class="radio"></i><b>${esc(t.name || t.id)}</b><i class="tag ${t.status === 'tested-mock' ? 'ok' : t.status === 'sdk-untested' ? 'warn' : 'grey'}">${esc(t.status || 'unknown')}</i></button>`).join('');
    qa('.st-target').forEach((b) => b.onclick = () => { st.target = b.dataset.id; qa('.st-target').forEach((x) => x.classList.toggle('on', x === b)); if (!st.running) setStep(6); });
  }
  async function loadTargets() {
    try {
      const r = await lab.transferTargets(); st.targets = (r && r.targets) || []; st.targetsError = null;
      if (st.targets.length && !st.targets.some((t) => t.id === st.target)) st.target = st.targets[0].id;
      renderTargets(st.targets.length ? null : 'no targets reported');
    } catch (e) { st.targetsError = e.message; renderTargets(e.message); }
  }
  function renderBundle(b, err) {
    const el = q('.st-bundle');
    if (err) { el.innerHTML = `<div class="st-err"><b>Export failed</b><span>${esc(err)}</span></div>`; return; }
    if (!b) { el.innerHTML = ''; return; }
    el.innerHTML = `<div class="st-bhead"><span class="mono" title="${esc(b.bundle)}">${esc(b.bundle)}</span><button class="st-btn st-reveal">Reveal in folder</button></div>
      ${(b.warnings || []).map((w) => `<div class="st-bwarn">${esc(w)}</div>`).join('')}
      <ul class="st-files mono">${(b.files || []).map((f) => `<li>${esc(f)}</li>`).join('')}</ul>
      ${b.zip ? `<div class="st-bzip mono" title="${esc(b.zip)}">zip: ${esc(b.zip)}</div>` : ''}
      ${b.validation ? `<div class="st-bzip mono ${b.validation.ok ? 'okc' : 'badc'}">validation: ${b.validation.ok ? 'ok' : 'FAILED'} · ${b.validation.n_checks ?? '?'} checks${(b.validation.errors || []).length ? ' · ' + esc(b.validation.errors.join('; ')) : ''}</div>` : ''}`;
    q('.st-reveal').onclick = async () => { try { await lab.transferReveal(b.zip || b.bundle); } catch (e) { log('reveal: ' + e.message, 'err'); } };
    if (!motion.reduce) gsap.from(el.querySelectorAll('.st-bhead, .st-bwarn, .st-files li'), { opacity: 0, x: -6, stagger: 0.025, duration: 0.25, clearProps: 'all' });
  }
  async function exportBundle() {
    const run = q('.f-drun').value; if (!run) { renderBundle(null, 'no source run - train a robot first'); return; }
    const btn = q('.st-export'); btn.disabled = true; btn.textContent = 'Exporting...';
    setStep(6); log(`export ${run} -> ${st.target}`, 'sys');
    try {
      const r = await lab.transferExport(run, st.target);
      st.bundle = r; renderBundle(r);
      log(`bundle ${r.bundle}: ${(r.files || []).length} files` + ((r.warnings || []).length ? ', ' + r.warnings.length + ' warning(s)' : ''), 'ok');
      (r.warnings || []).forEach((w) => log('warning: ' + w, 'warn'));
    } catch (e) { st.bundle = null; renderBundle(null, e.message); log('export: ' + e.message, 'err'); }
    finally { btn.textContent = 'Export bundle'; syncButtons(); }
  }
  async function dryRunBundle(neps) {
    neps = neps || 3;
    if (!st.bundle) return;
    const run = st.runs.find((r) => q('.f-drun').value === 'user_runs/' + r.run_id);
    if (run && run.robot && st.robots.some((x) => x.name === run.robot) && (!st.sel || st.sel.name !== run.robot)) await selectRobot(run.robot);
    hideSummary(); st.episodes = []; st.nEp = neps; st.lastFrame = null; st.spikes = null; resetTrail(); drawCurve(); drawHeat(); setProgress(0, neps);
    st.running = true; st.mode = 'deploy'; syncButtons();
    try {
      await lab.transferRun(st.bundle.bundle, neps);
      setStep(6); setStatus('run', 'Closed-loop dry run'); log('closed-loop dry run against mock MEA (simulated culture)', 'sys');
    } catch (e) { st.running = false; syncButtons(); setStatus('err', 'Error'); log('dry run: ' + e.message, 'err'); trainErr(e.message); }
  }
  q('.st-export').onclick = exportBundle;
  q('.st-dryrun').onclick = dryRunBundle;
  fillDeployRuns(); renderTargets(); loadTargets();

  // ---------- boot ----------
  renderLib(); renderRobotProps(); setStep(1); setStatus('idle', 'Idle'); drawCurve(); drawHeat(); renderJobs(); syncButtons();
  log('BNN Robot Studio ready - controllers are simulated Izhikevich cultures (no living neurons).', 'sys');
  const ready = Promise.all([refreshList(), refreshRuns()]);

  return {
    setActive(on) {
      const was = st.active; st.active = on;
      if (on && !was) { resize(); last = performance.now(); requestAnimationFrame(frame); enter(); }
    },
    ready: () => ready,
    whenScene: async (ms = 60000) => { const t0 = Date.now(); while (st.sceneLoading && Date.now() - t0 < ms) await new Promise((r) => setTimeout(r, 200)); },
    status: () => ({
      active: st.active, robots: st.robots.length, robotNames: st.robots.map((r) => r.name), listError: st.listError,
      selected: st.sel && st.sel.name, sceneBodies: st.sceneBodies || 0, sceneGeoms: st.sceneGeoms || 0, sceneError: st.sceneError,
      step: st.step, running: st.running, events: st.events, frames: st.frames, episodes: st.episodes.length, done: st.done,
      jobs: st.jobs.length, runs: st.runs.length, targets: st.targets.length, targetsError: st.targetsError || null, bundle: st.bundle ? st.bundle.bundle : null, mode: st.mode || null, runsError: st.runsError, lastError: st.lastError, fps: Math.round(fps),
    }),
    select: (name) => selectRobot(name),
    exportAndDryRun: async (target, neps) => {
      fillDeployRuns(); if (target) st.target = target; renderTargets();
      await exportBundle();
      if (!st.bundle) return { exported: false };
      const done = new Promise((res) => { lab.onStudioEvent((o) => { if (o.type === 'exit') res(o); }); setTimeout(() => res({ timeout: true }), 150000); });
      await dryRunBundle(neps);
      return { exported: true, files: (st.bundle.files || []).length, exit: await done };
    },
    focus: (sec) => { const s = q(`.st-grp[data-sec="${String(sec).replace(/[^a-z]/g, '')}"]`); if (s) s.scrollIntoView({ block: 'start' }); },
    train: (overrides) => { Object.entries(overrides || {}).forEach(([k, v]) => { if (k === 'platform') setPlatform(v); else if (k === 'dryRun') q('.f-dry').checked = !!v; else if (k === 'episodes') q('.f-episodes').value = v; }); syncButtons(); return train(); },
    importFile,
    demo: {
      el: (s) => q(s),
      setStep, showTab, refreshJobs, refreshRuns, replayRun, setPlatform: (p) => { setPlatform(p); syncButtons(); }, flash, log,
      setStatus, exportBundle: async (target) => { fillDeployRuns(); if (target) st.target = target; renderTargets(); await exportBundle(); return st.bundle; },
      dryRun: (n) => dryRunBundle(n), stop: (slot) => lab.studioStop(slot), hideSummary,
      state: () => st,
      focus: (sec) => { const s = q(`.st-grp[data-sec="${sec}"]`); if (s) { s.scrollIntoView({ block: 'nearest', behavior: motion.reduce ? 'auto' : 'smooth' }); flash(s); } },
      // animate a property-grid field to a new value (numbers count, others snap), with a highlight
      setField: (cls, v) => new Promise((res) => {
        const el = q('.' + cls); if (!el) return res();
        const row = el.closest('.pg'); flash(row);
        if (el.type === 'number' && !motion.reduce) {
          const o = { v: Number(el.value) || 0 }; const dec = String(v).includes('.') ? String(v).split('.')[1].length : 0;
          gsap.to(o, { v: Number(v), duration: 0.7, ease: 'power2.inOut', onUpdate: () => { el.value = o.v.toFixed(dec); }, onComplete: () => { el.value = v; res(); } });
        } else { el.value = v; res(); }
      }),
      setRule: (r) => { const b = q(`.f-rule button[data-v="${r}"]`); if (b) { b.click(); flash(b.closest('.pg')); } },
      setVms: (n) => { const r = q('.f-vms'); r.value = n; r.dispatchEvent(new Event('input')); flash(r.closest('.pg')); },
      setDry: (on) => { q('.f-dry').checked = !!on; syncButtons(); },
      setEpisodes: (n) => { q('.f-episodes').value = n; flash(q('.f-episodes').closest('.pg')); },
      pressTrain: () => { const b = q('.st-train'); if (!motion.reduce) gsap.fromTo(b, { scale: 0.95 }, { scale: 1, duration: 0.25, clearProps: 'transform' }); return train(); },
      select: (name) => selectRobot(name),
    },
  };
}
