// Surgical tab: SimReady robotic_system_a01 cart (CC-BY-4.0) assembled from app/assets/surgical/tree.json + per-body GLBs,
// forward kinematics from joint angles, arm-1 sliders, target marker, and replay of surgical/results/trajectory.json.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { parseLoader } from './simready.js';

const HOME = { joint_slider_01_joint_00: -0.08, joint_junction_01_joint_00: 0, joint_elbowBig_01_joint_00: 0, joint_elbowSmall_01_joint_00: 0,
  joint_elbowLong_01_joint_00: 0.25, joint_elbowMiddle_01_joint_00: 0.25, joint_elbowShort_01_joint_00: -0.1, joint_mechanism_01_joint_00: 0 }; // surgical/arm.py HOME_Q
const TABLE_Z = 0.70; // surgical/arm.py virtual patient-table top (m)
const short = (n) => n.replace(/^joint_/, '').replace(/_joint_00$/, '');

export function initSurgical(root, lab, opts = {}) {
  root.innerHTML = `<div class="surg">
    <div class="sviz"><canvas></canvas><div class="sbadge">Simulated culture model - no living neurons</div>
      <div class="shud"></div><div class="stag"></div>
      <div class="credits" style="bottom:8px">Robot: robotic_system_a01 (CC-BY-4.0)</div></div>
    <div class="spanel panel">
      <h2>Arm 1 joints (forward kinematics)</h2><div class="sliders"></div>
      <button class="shome">Home pose</button>
      <h2>Instrument tip / target</h2><div class="sread mono"></div>
      <h2>Replay</h2>
      <div class="srow"><button class="splay">Play replay</button><select class="sep"></select></div>
      <input class="sscrub" type="range" min="0" max="0" value="0" style="width:100%;padding:0">
      <div class="spos mono dim"></div>
      <h2>Culture electrodes (8x8 spike counts, this step)</h2><canvas class="sheat" width="200" height="200"></canvas>
      <div class="dim" style="font-size:10.5px;margin-top:6px">Replay source: surgical/results/trajectory.json (written by surgical/record_and_render.py). Without it a clearly labelled demo animation runs instead.</div>
    </div></div>`;
  const q = (s) => root.querySelector(s);
  const canvas = q('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x050607);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 60);
  camera.position.set(2.4, 2.1, 2.9);
  const controls = new OrbitControls(camera, canvas); controls.target.set(0, 0.95, 0.3); controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495;
  scene.add(new THREE.HemisphereLight(0xdfe8f0, 0x101214, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(2, 5, 3); scene.add(key);
  const fill = new THREE.DirectionalLight(0x9fc4ff, 0.5); fill.position.set(-3, 2, -2); scene.add(fill);
  const lamp = new THREE.SpotLight(0xffffff, 18, 6, 0.45, 0.5); lamp.position.set(0.6, 3.2, 0.6); scene.add(lamp, lamp.target);
  // clinical bay: dark floor with faint grid, back walls
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), new THREE.MeshStandardMaterial({ color: 0x0c0e11, roughness: 0.35, metalness: 0.2 }));
  floor.rotation.x = -Math.PI / 2; scene.add(floor);
  const grid = new THREE.GridHelper(14, 28, 0x1c232b, 0x141a20); grid.position.y = 0.001; scene.add(grid);
  const wallM = new THREE.MeshStandardMaterial({ color: 0x0f1318, roughness: 0.9 });
  const w1 = new THREE.Mesh(new THREE.PlaneGeometry(14, 4), wallM); w1.position.set(0, 2, -5); scene.add(w1);
  const w2 = new THREE.Mesh(new THREE.PlaneGeometry(14, 4), wallM); w2.position.set(-5, 2, 0); w2.rotation.y = Math.PI / 2; scene.add(w2);
  const mj = new THREE.Group(); mj.rotation.x = -Math.PI / 2; scene.add(mj); // children use MuJoCo Z-up metres directly

  const st = { tree: null, nodes: {}, q: {}, loaded: 0, active: false, tipMJ: new THREE.Vector3(), replay: null, demo: false, k: 0, playing: false, ep: 0, err: null };
  const tipBall = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd23a, toneMapped: false }));
  const target = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12), new THREE.MeshBasicMaterial({ color: 0x76b900, transparent: true, opacity: 0.9 }));
  const targetRing = new THREE.Mesh(new THREE.RingGeometry(0.022, 0.028, 32), new THREE.MeshBasicMaterial({ color: 0x76b900, side: THREE.DoubleSide }));
  target.add(targetRing); mj.add(target);
  const TRAIL_N = 2000, trailPos = new Float32Array(TRAIL_N * 3);
  const trailGeo = new THREE.BufferGeometry(); trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3)); trailGeo.setDrawRange(0, 0);
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffd23a })); trail.frustumCulled = false; mj.add(trail);

  const matFor = (n) => {
    if (/^wheel/.test(n)) return new THREE.MeshStandardMaterial({ color: 0x1a1c1f, roughness: 0.8 });
    if (/^(mechanism|knob)/.test(n)) return new THREE.MeshStandardMaterial({ color: 0x3a4048, roughness: 0.45, metalness: 0.5 });
    if (/^(junction|slider)/.test(n)) return new THREE.MeshStandardMaterial({ color: 0xb9c0c8, roughness: 0.35, metalness: 0.4 });
    return new THREE.MeshStandardMaterial({ color: 0xeef1f4, roughness: 0.32, metalness: 0.08 });
  };

  // ---- forward kinematics: body = T(pos) R(quat) * J(q); hinge J = T(jp) R(axis, q-ref) T(-jp); slide J = T(axis (q-ref)) ----
  const _m = new THREE.Matrix4(), _j = new THREE.Matrix4(), _t = new THREE.Matrix4(), _ax = new THREE.Vector3(), _qq = new THREE.Quaternion();
  function fk(qmap) {
    if (!st.tree) return;
    for (const b of st.tree.bodies) {
      const nd = st.nodes[b.name]; if (!nd) continue;
      nd.matrix.compose(new THREE.Vector3(...b.pos), _qq.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]).normalize(), new THREE.Vector3(1, 1, 1));
      const J = b.joint;
      if (J) {
        const v = (qmap[J.name] ?? J.ref) - J.ref;
        _ax.set(...J.axis).normalize();
        if (J.type === 'slide') _j.makeTranslation(_ax.x * v, _ax.y * v, _ax.z * v);
        else if (J.type === 'hinge') { _j.makeTranslation(...J.pos).multiply(_m.makeRotationAxis(_ax, v)).multiply(_t.makeTranslation(-J.pos[0], -J.pos[1], -J.pos[2])); }
        else _j.identity();
        nd.matrix.multiply(_j);
      }
      nd.matrixWorldNeedsUpdate = true;
    }
    mj.updateMatrixWorld(true);
    tipBall.getWorldPosition(st.tipMJ); mj.worldToLocal(st.tipMJ);
  }

  async function load() {
    try {
      st.tree = JSON.parse(new TextDecoder().decode(await lab.readAsset('surgical/tree.json')));
      for (const b of st.tree.bodies) {
        const g = new THREE.Group(); g.name = b.name; g.matrixAutoUpdate = false; st.nodes[b.name] = g;
        (b.parent ? st.nodes[b.parent] : mj).add(g);
      }
      st.nodes[st.tree.tip_body].add(tipBall); tipBall.position.set(...st.tree.tip_local);
      tipBall.updateMatrix();
      await Promise.all(st.tree.bodies.filter((b) => b.mesh).map(async (b) => {
        const gl = await parseLoader(lab, 'surgical/' + b.name + '.glb');
        gl.scene.traverse((o) => { if (o.isMesh) { if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals(); o.material = matFor(b.name); } });
        if (b.geom_pos) { gl.scene.position.set(...b.geom_pos); gl.scene.quaternion.set(b.geom_quat[1], b.geom_quat[2], b.geom_quat[3], b.geom_quat[0]); }
        st.nodes[b.name].add(gl.scene); st.loaded++;
      }));
      for (const b of st.tree.bodies) if (b.joint) st.q[b.joint.name] = HOME[b.joint.name] ?? b.joint.ref;
      fk(st.q);
      // patient table under the instrument at the home pose, top at TABLE_Z
      const tb = new THREE.Mesh(new THREE.BoxGeometry(0.62, 1.9, 0.06), new THREE.MeshStandardMaterial({ color: 0x2b3a44, roughness: 0.5 }));
      tb.position.set(st.tipMJ.x, st.tipMJ.y, TABLE_Z - 0.03); mj.add(tb);
      const ped = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.5, TABLE_Z - 0.06), new THREE.MeshStandardMaterial({ color: 0x1a2026, roughness: 0.6, metalness: 0.3 }));
      ped.position.set(st.tipMJ.x, st.tipMJ.y, (TABLE_Z - 0.06) / 2); mj.add(ped);
      const drape = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.9, 0.02), new THREE.MeshStandardMaterial({ color: 0x2f7d8c, roughness: 0.9 }));
      drape.position.set(st.tipMJ.x, st.tipMJ.y, TABLE_Z + 0.01); mj.add(drape);
      target.position.set(st.tipMJ.x + 0.04, st.tipMJ.y - 0.05, Math.max(TABLE_Z + 0.06, st.tipMJ.z - 0.08));
      lamp.target.position.set(st.tipMJ.x, TABLE_Z, -st.tipMJ.y); controls.target.set(st.tipMJ.x * 0.5, 0.95, -st.tipMJ.y * 0.5);
      buildSliders(); readout(); drawHeat([]);
    } catch (e) { st.err = String(e.message || e); console.warn('surgical load failed: ' + st.err); q('.stag').textContent = 'robot assets missing: run app/tools/build_surgical_glb.py'; }
  }

  function buildSliders() {
    const box = q('.sliders'); box.innerHTML = '';
    for (const jn of st.tree.arm1_joints) {
      const b = st.tree.bodies.find((x) => x.joint && x.joint.name === jn); if (!b) continue;
      const [lo, hi] = b.joint.range || [-Math.PI, Math.PI], slide = b.joint.type === 'slide';
      const row = document.createElement('label'); row.className = 'sj';
      row.innerHTML = `<span>${short(jn)}</span><input type="range" min="${lo}" max="${hi}" step="${(hi - lo) / 400}" value="${st.q[jn]}"><em class="mono"></em>`;
      const inp = row.querySelector('input'), em = row.querySelector('em');
      const show = () => { em.textContent = slide ? (st.q[jn] * 1000).toFixed(0) + ' mm' : (st.q[jn] * 57.2958).toFixed(1) + '°'; };
      inp.oninput = () => { st.playing = false; q('.splay').textContent = 'Play replay'; st.q[jn] = Number(inp.value); fk(st.q); show(); readout(); };
      row.sync = () => { inp.value = st.q[jn]; show(); };
      show(); box.appendChild(row);
    }
  }
  const syncSliders = () => q('.sliders').querySelectorAll('.sj').forEach((r) => r.sync && r.sync());
  function readout() {
    const t = st.tipMJ, g = target.position, d = t.distanceTo(g) * 1000;
    q('.sread').innerHTML = `tip    ${t.x.toFixed(3)} ${t.y.toFixed(3)} ${t.z.toFixed(3)} m<br>target ${g.x.toFixed(3)} ${g.y.toFixed(3)} ${g.z.toFixed(3)} m<br>error  <b style="color:${d < 10 ? 'var(--acc)' : '#fff'}">${d.toFixed(1)} mm</b>`;
  }
  q('.shome').onclick = () => { for (const k in HOME) st.q[k] = HOME[k]; st.q['joint_handBase_01_joint_00'] = 0; fk(st.q); syncSliders(); readout(); };

  // ---- replay ----
  function normEpisodes(d) { // accepts {joints, episodes:[{steps}]}, {steps}, or a bare array of steps
    const eps = Array.isArray(d) ? [{ steps: d }] : d.episodes || (d.steps ? [{ steps: d.steps }] : []);
    const joints = d.joints || (st.tree ? st.tree.arm1_joints.slice(1) : []);
    return eps.filter((e) => e.steps && e.steps.length).map((e, i) => ({ name: 'episode ' + i + (e.target_index != null ? ' (target ' + e.target_index + ')' : ''), result: e.result, steps: e.steps.map((s) => {
      let qm = {};
      if (Array.isArray(s.q)) s.q.forEach((v, k) => { if (joints[k]) qm[joints[k]] = v; });
      else if (s.q && typeof s.q === 'object') qm = s.q; else if (s.joints && typeof s.joints === 'object') qm = s.joints;
      return { t: s.t, q: qm, tip: s.tip || s.tip_pos, target: s.target || s.target_pos, dist: s.dist_mm, spikes: s.spike_counts || s.spikes || s.electrode_counts };
    }) }));
  }
  function demoReplay() {
    const eps = [];
    for (let e = 0; e < 3; e++) {
      const steps = [], tg = [st.tipMJ.x + 0.05 * Math.cos(e * 2.1), st.tipMJ.y + 0.05 * Math.sin(e * 2.1), st.tipMJ.z - 0.04 - 0.02 * e];
      for (let i = 0; i < 120; i++) {
        const u = 1 - Math.exp(-i / 25), qm = {};
        for (const k in HOME) qm[k] = HOME[k];
        qm.joint_elbowBig_01_joint_00 = 0.25 * u * Math.cos(e * 2.1); qm.joint_elbowSmall_01_joint_00 = 0.2 * u * Math.sin(e * 2.1 + 1);
        qm.joint_elbowLong_01_joint_00 = 0.25 + 0.15 * u; qm.joint_elbowMiddle_01_joint_00 = 0.25 + 0.1 * u * (e - 1);
        const spikes = Array.from({ length: 64 }, (_, ch) => Math.round(Math.max(0, (ch < 14 ? 6 * (1 - u) : ch >= 40 ? 4 * Math.abs(Math.sin(i / 6 + ch)) : 1) + Math.random() * 2)));
        steps.push({ t: i * 0.05, q: qm, tip: null, target: tg, spikes });
      }
      eps.push({ name: 'demo episode ' + e, steps });
    }
    return eps;
  }
  async function loadReplay() {
    let info = null; st.replay = null;
    try { info = (await lab.listReplays()).surgical; } catch (e) {}
    if (info) { try { st.replay = normEpisodes(await lab.loadReplay('surgical')); st.demo = false; } catch (e) { console.warn('surgical replay unreadable: ' + e.message); } }
    if (!st.replay || !st.replay.length) { st.replay = demoReplay(); st.demo = true; }
    q('.sep').innerHTML = st.replay.map((e, i) => `<option value="${i}">${e.name}</option>`).join('');
    q('.stag').innerHTML = st.demo ? '<span class="mock">DEMO ANIMATION</span> surgical/results/trajectory.json not found: synthetic joint motion' : '<b>Replay</b> surgical/results/trajectory.json · simulated-culture control of arm 1';
    setEp(0);
  }
  function setEp(i) { st.ep = i; st.k = 0; trailGeo.setDrawRange(0, 0); q('.sscrub').max = st.replay[i].steps.length - 1; q('.sscrub').value = 0; }
  q('.sep').onchange = () => setEp(Number(q('.sep').value));
  q('.sscrub').oninput = () => { st.k = Number(q('.sscrub').value); rebuildTrail(); };
  function startReplay() { const go = () => { st.playing = true; q('.splay').textContent = 'Pause replay'; }; if (!st.replay) return loadReplay().then(go); go(); return Promise.resolve(); }
  q('.splay').onclick = () => { if (st.playing) { st.playing = false; q('.splay').textContent = 'Play replay'; } else startReplay(); };
  function rebuildTrail() {
    const s = st.replay[st.ep].steps, n = Math.min(TRAIL_N, Math.floor(st.k) + 1);
    for (let i = 0; i < n; i++) { applyStep(s[i], true); trailPos.set([st.tipMJ.x, st.tipMJ.y, st.tipMJ.z], i * 3); }
    trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, n);
  }
  function applyStep(s, quiet) {
    Object.assign(st.q, s.q); fk(st.q);
    if (s.tip && !quiet) st.tipErr = st.tipMJ.distanceTo(new THREE.Vector3(...s.tip)) * 1000; // FK vs recorded tip (mm)
    if (s.target) target.position.set(...s.target);
  }
  function drawHeat(sp) {
    const cv = q('.sheat'), g = cv.getContext('2d'), c = cv.width / 8;
    g.fillStyle = '#050505'; g.fillRect(0, 0, cv.width, cv.height);
    const a = Array.isArray(sp) ? sp : [], mx = Math.max(1, ...a);
    for (let ch = 0; ch < 64; ch++) {
      const v = a[ch] ?? 0, f = Math.min(1, v / mx);
      g.fillStyle = a.length > ch ? `rgb(${Math.round(20 + 235 * f)},${Math.round(30 + 180 * f * f)},${Math.round(40 + 40 * (1 - f))})` : '#101010';
      g.fillRect((ch % 8) * c + 1, (ch >> 3) * c + 1, c - 2, c - 2);
      if (v) { g.fillStyle = f > 0.6 ? '#000' : '#bbb'; g.font = '10px Consolas'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(String(v), (ch % 8) * c + c / 2, (ch >> 3) * c + c / 2); }
    }
  }

  function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(q('.sviz'));
  let last = performance.now(), frames = 0, fpsT = 0;
  function frame(now) {
    if (!st.active) return;
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (st.playing && st.replay) {
      const s = st.replay[st.ep].steps, k0 = Math.floor(st.k);
      st.k += dt / 0.05;
      if (st.k >= s.length + 20) { st.k = 0; trailGeo.setDrawRange(0, 0); }
      const i = Math.min(s.length - 1, Math.floor(st.k)), cur = s[i];
      applyStep(cur);
      if (i !== k0 || i === 0) { if (i < TRAIL_N) { trailPos.set([st.tipMJ.x, st.tipMJ.y, st.tipMJ.z], i * 3); trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, i + 1); } }
      q('.sscrub').value = i; drawHeat(cur.spikes); syncSliders(); readout();
      q('.spos').textContent = `step ${i + 1}/${s.length} · t=${(cur.t ?? i * 0.05).toFixed(2)} s${cur.dist != null ? ' · dist ' + cur.dist.toFixed(1) + ' mm' : ''}${st.tipErr != null && !st.demo ? ' · FK-vs-recorded tip ' + st.tipErr.toFixed(1) + ' mm' : ''}`;
    } else if (st.replay && st.k >= 0) { const s = st.replay[st.ep].steps, i = Math.min(s.length - 1, Math.floor(st.k)); drawHeat(s[i].spikes); }
    targetRing.lookAt(camera.position.clone().applyMatrix4(target.parent.matrixWorld.clone().invert()));
    controls.update(); renderer.render(scene, camera);
    frames++; fpsT += dt; if (fpsT > 0.5) { q('.shud').textContent = `${(frames / fpsT).toFixed(0)} fps · ${st.loaded}/${st.tree ? st.tree.bodies.length : 0} bodies · drag to orbit, wheel to zoom`; frames = 0; fpsT = 0; }
  }
  const ready = load();
  return {
    setActive(on) { const was = st.active; st.active = on; if (on && !was) { resize(); last = performance.now(); requestAnimationFrame(frame); } },
    refresh() { if (st.replay) { const p = st.playing; loadReplay().then(() => { if (p) startReplay(); }); } },
    startReplay: () => ready.then(startReplay),
    status: () => ({ bodies: st.tree ? st.tree.bodies.length : 0, loaded: st.loaded, err: st.err, tip: st.tipMJ.toArray().map((v) => +v.toFixed(4)),
      replay: st.replay ? (st.demo ? 'demo' : 'surgical/results/trajectory.json') : null, step: Math.floor(st.k), playing: st.playing, tipErrMm: st.tipErr ?? null }),
  };
}
