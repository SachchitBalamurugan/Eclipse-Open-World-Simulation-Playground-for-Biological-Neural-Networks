import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { PARTS, PHYS } from './parts.js';
import { initFleet } from './fleet.js';

const $ = (id) => document.getElementById(id);
const lab = window.lab;
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const QS = new URLSearchParams(location.search);
const SELFTEST = QS.get('selftest') === '1';
const START_CAM = ['dish', 'robot', 'overview'].includes(QS.get('camera')) ? QS.get('camera') : 'overview';
const START_TAB = QS.get('tab') || 'simulate';
const DEFAULT_RUNS = ['culture_teach_s0', 'culture_reservoir_s0'];

// ---------- seeded rng ----------
function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rnd = mulberry(1024);

// ---------- zones ----------
const ZC = { sens: 0x4da3ff, dep: 0xffb03a, hold: 0x36d6c3, fb: 0xff5fa8, rew: 0xa47bff, inter: 0x7f93b3 };
const zoneOf = (ch) => ch < 16 ? 'sens' : ch >= 40 && ch < 48 ? 'dep' : ch >= 48 && ch < 56 ? 'hold' : ch >= 56 && ch < 60 ? 'fb' : ch >= 60 ? 'rew' : 'inter';
const STIM_COL = [0x4da3ff, 0xff5fa8, 0xa47bff, 0x5dff8a];
const ex = (ch) => (ch % 8) - 3.5, ez = (ch) => (ch >> 3) - 3.5;

// ---------- three setup ----------
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030303);
scene.fog = new THREE.Fog(0x030303, 30, 70);
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.49; controls.minDistance = 3; controls.maxDistance = 45;
scene.add(new THREE.HemisphereLight(0x9bb8ff, 0x101820, 0.9));
const sun = new THREE.DirectionalLight(0xdfe8ff, 1.3); sun.position.set(6, 12, 8); scene.add(sun);

const FX = 11.5; // field centre x
const CAMS = { dish: [[0, 7.2, 9.2], [0, 0.3, 0]], robot: [[FX - 3.4 + 1.05, 0.75, 2.5 + 1.25], [FX - 3.4, 0.28, 2.5]], overview: [[5.6, 12.5, 19], [5.6, 0, 0.5]] };
let camTween = null, camMode = 'overview';
function setCam(name, instant) {
  const p = CAMS[name][0].slice(), t = CAMS[name][1].slice();
  camMode = name;
  if (name === 'robot' && typeof rover !== 'undefined') { const dx = rover.x - HOME; p[0] += dx; t[0] += dx; }
  document.querySelectorAll('#cams [data-cam]').forEach((b) => b.classList.toggle('on', b.dataset.cam === name));
  if (instant || REDUCED) { camera.position.set(...p); controls.target.set(...t); return; }
  camTween = { p0: camera.position.clone(), t0: controls.target.clone(), p1: new THREE.Vector3(...p), t1: new THREE.Vector3(...t), k: 0 };
}
document.querySelectorAll('#cams [data-cam]').forEach((b) => b.onclick = () => setCam(b.dataset.cam));

let composer = null, bloomOn = !REDUCED;
try {
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(512, 512), 0.75, 0.5, 0.55));
  composer.addPass(new OutputPass());
} catch (e) { console.warn('bloom unavailable', e); composer = null; bloomOn = false; }
$('bloomBtn').classList.toggle('on', bloomOn);
$('bloomBtn').onclick = () => { bloomOn = !bloomOn && !!composer; $('bloomBtn').classList.toggle('on', bloomOn); };

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); if (composer) composer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($('centre')); resize();

function label(text, color, size) {
  const c = document.createElement('canvas'); c.width = 256; c.height = 64;
  const g = c.getContext('2d'); g.font = '600 30px Segoe UI, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillStyle = color; g.fillText(text, 128, 32);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  s.scale.set(size * 4, size, 1); return s;
}

// ---------- dish ----------
const dish = new THREE.Group(); scene.add(dish);
{
  const floor = new THREE.Mesh(new THREE.CylinderGeometry(6.2, 6.2, 0.2, 64), new THREE.MeshStandardMaterial({ color: 0x0b111b, roughness: 0.5, metalness: 0.3 }));
  floor.position.y = -0.12; dish.add(floor);
  const wall = new THREE.Mesh(new THREE.CylinderGeometry(6.2, 6.2, 1.7, 64, 1, true), new THREE.MeshStandardMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0.1, side: THREE.DoubleSide, roughness: 0.1, depthWrite: false }));
  wall.position.y = 0.75; dish.add(wall);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(6.2, 0.05, 8, 96), new THREE.MeshBasicMaterial({ color: 0x5f8fb8 }));
  rim.rotation.x = Math.PI / 2; rim.position.y = 1.6; dish.add(rim);
  const chip = new THREE.Mesh(new THREE.BoxGeometry(8.6, 0.04, 8.6), new THREE.MeshStandardMaterial({ color: 0x101a28, roughness: 0.4, metalness: 0.5 }));
  chip.position.y = 0; dish.add(chip);
  const zone = (x, z, w, d, col) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.13, depthWrite: false })); m.rotation.x = -Math.PI / 2; m.position.set(x, 0.03, z); dish.add(m); };
  zone(0, -3, 8, 2, ZC.sens); zone(0, 1.5, 8, 1, ZC.dep); zone(0, 2.5, 8, 1, ZC.hold); zone(-2, 3.5, 4, 1, ZC.fb); zone(2, 3.5, 4, 1, ZC.rew);
  const pads = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.2, 0.2, 0.03, 16), new THREE.MeshBasicMaterial(), 64);
  const m4 = new THREE.Matrix4(), c = new THREE.Color();
  for (let ch = 0; ch < 64; ch++) { m4.setPosition(ex(ch), 0.045, ez(ch)); pads.setMatrixAt(ch, m4); pads.setColorAt(ch, c.setHex(ZC[zoneOf(ch)]).multiplyScalar(0.45)); }
  dish.add(pads);
  const L = (t, col, x, z) => { const s = label(t, col, 0.5); s.position.set(x, 0.5, z); dish.add(s); };
  L('SENSORY', '#4da3ff', -5.3, -3); L('DEPLOY', '#ffb03a', -5.3, 1.5); L('HOLD', '#36d6c3', -5.3, 2.5); L('FB / REWARD', '#ff5fa8', -5.5, 3.6);
}

// neurons
const N = 1024;
const npos = new Float32Array(N * 3), nbase = new Float32Array(N * 3), nhot = new Float32Array(N * 3), energy = new Float32Array(N);
const neurons = new THREE.InstancedMesh(new THREE.SphereGeometry(0.06, 8, 6), new THREE.MeshBasicMaterial({ toneMapped: false }), N);
{
  const m4 = new THREE.Matrix4(), c = new THREE.Color(), w = new THREE.Color(0xffffff);
  for (let i = 0; i < N; i++) {
    const ch = i >> 4;
    const x = ex(ch) + (rnd() - 0.5) * 0.86, y = 0.14 + rnd() * 0.75, z = ez(ch) + (rnd() - 0.5) * 0.86;
    npos.set([x, y, z], i * 3); m4.setPosition(x, y, z); neurons.setMatrixAt(i, m4);
    c.setHex(ZC[zoneOf(ch)]); nbase.set([c.r * 0.3, c.g * 0.3, c.b * 0.3], i * 3);
    c.lerp(w, 0.35); nhot.set([c.r * 2.2, c.g * 2.2, c.b * 2.2], i * 3);
    neurons.setColorAt(i, c.setRGB(nbase[i * 3], nbase[i * 3 + 1], nbase[i * 3 + 2]));
  }
  dish.add(neurons);
}
// neurites: local links + sensory->motor bundle
const bundle = []; // {pts:Float32Array(3*(S+1)), ch}
{
  const v = [];
  for (let i = 0; i < N; i++) {
    if (rnd() > 0.55) continue;
    const ch = i >> 4, dc = Math.floor(rnd() * 3) - 1, dr = Math.floor(rnd() * 3) - 1;
    const c2 = Math.min(7, Math.max(0, (ch % 8) + dc)) + 8 * Math.min(7, Math.max(0, (ch >> 3) + dr));
    const j = c2 * 16 + Math.floor(rnd() * 16);
    v.push(npos[i * 3], npos[i * 3 + 1], npos[i * 3 + 2], npos[j * 3], npos[j * 3 + 1], npos[j * 3 + 2]);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  dish.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x5f7fa8, transparent: true, opacity: 0.16, depthWrite: false })));
  const bv = [], S = 14;
  for (let k = 0; k < 110; k++) {
    const a = Math.floor(rnd() * 256), b = 640 + Math.floor(rnd() * 256);
    const A = new THREE.Vector3(npos[a * 3], npos[a * 3 + 1], npos[a * 3 + 2]), B = new THREE.Vector3(npos[b * 3], npos[b * 3 + 1], npos[b * 3 + 2]);
    const M = A.clone().lerp(B, 0.5); M.y = 1.05 + rnd() * 0.45; M.x = M.x * 0.45;
    const curve = new THREE.QuadraticBezierCurve3(A, M, B), pts = new Float32Array((S + 1) * 3);
    for (let s = 0; s <= S; s++) { const p = curve.getPoint(s / S); pts.set([p.x, p.y, p.z], s * 3); if (s > 0) bv.push(pts[(s - 1) * 3], pts[(s - 1) * 3 + 1], pts[(s - 1) * 3 + 2], p.x, p.y, p.z); }
    bundle.push({ pts, ch: a >> 4, S });
  }
  const bg = new THREE.BufferGeometry(); bg.setAttribute('position', new THREE.Float32BufferAttribute(bv, 3));
  dish.add(new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0x6fb6ff, transparent: true, opacity: 0.2, depthWrite: false })));
}
const bundleByCh = Array.from({ length: 16 }, () => []);
bundle.forEach((b, i) => bundleByCh[b.ch].push(i));
// travelling pulses
const NP = 160, pulse = { line: new Int16Array(NP).fill(-1), k: new Float32Array(NP), next: 0 };
const pulseGeo = new THREE.BufferGeometry();
pulseGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NP * 3).fill(999), 3));
dish.add(new THREE.Points(pulseGeo, new THREE.PointsMaterial({ color: 0xbfe4ff, size: 0.17, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })));
// stim rings + beams
const NR = 56, rings = [];
{
  const rg = new THREE.RingGeometry(0.24, 0.32, 24), bgm = new THREE.CylinderGeometry(0.035, 0.035, 1.5, 6);
  for (let i = 0; i < NR; i++) {
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, toneMapped: false, blending: THREE.AdditiveBlending });
    const ring = new THREE.Mesh(rg, mat); ring.rotation.x = -Math.PI / 2;
    const beam = new THREE.Mesh(bgm, mat); beam.position.y = 0.75;
    const grp = new THREE.Group(); grp.add(ring, beam); grp.visible = false; dish.add(grp);
    rings.push({ grp, ring, mat, life: 0 });
  }
}
let ringNext = 0;
function fireStim(ch, kind) {
  const r = rings[ringNext]; ringNext = (ringNext + 1) % NR;
  r.grp.position.set(ex(ch), 0.07, ez(ch)); r.mat.color.setHex(STIM_COL[kind] || STIM_COL[1]); r.life = 1; r.grp.visible = true;
}
function fireSpike(ch) {
  const i = ch * 16 + (Math.random() * 16 | 0);
  energy[i] = 1;
  if (ch < 16 && Math.random() < 0.1 && bundleByCh[ch].length) {
    const ls = bundleByCh[ch]; pulse.line[pulse.next] = ls[Math.random() * ls.length | 0]; pulse.k[pulse.next] = 0; pulse.next = (pulse.next + 1) % NP;
  }
}

// ---------- component-built rover (see app/parts.js, hardware/BOM.md) ----------
const ROVER_Z = 2.5;
const rollPos = new THREE.Vector3(-3.7, 0.3, ROVER_Z); // world-ish (field-local) position of the cover roll
const partMeshes = []; let rv = null;
function buildRover() {
  const grp = new THREE.Group();
  const M = (c, o) => new THREE.MeshStandardMaterial(Object.assign({ color: c, roughness: 0.55, metalness: 0.2 }, o || {}));
  const alu = M(0xb9c1cb, { metalness: 0.75, roughness: 0.35 }), black = M(0x16181b, { roughness: 0.9 }), dark = M(0x2a2e34);
  const add = (geo, mat, part, x, y, z, parent) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.userData.part = part; (parent || grp).add(m); partMeshes.push(m); return m; };
  const B = (x, y, z) => new THREE.BoxGeometry(x, y, z);
  const deckY = 0.107;
  // chassis: 2x 600 mm rails, 3x 400 mm cross members (2020 extrusion), 3 mm deck
  for (const z of [-0.19, 0.19]) add(B(0.6, 0.02, 0.02), alu, 'rail', 0, 0.095, z);
  for (const x of [-0.28, 0, 0.28]) add(B(0.02, 0.02, 0.36), alu, 'rail', x, 0.095, 0);
  add(B(0.6, 0.003, 0.4), M(0x8d949c, { metalness: 0.6, roughness: 0.4 }), 'deck', 0, deckY, 0);
  // 4x Pololu 37D gearmotors with encoders + Wild Thumper 120 mm wheels
  const wg = new THREE.CylinderGeometry(PHYS.wheelR, PHYS.wheelR, 0.06, 14); wg.rotateX(Math.PI / 2);
  const mg = new THREE.CylinderGeometry(0.0185, 0.0185, 0.12, 12); mg.rotateX(Math.PI / 2);
  const hg = new THREE.CylinderGeometry(0.022, 0.022, 0.065, 10); hg.rotateX(Math.PI / 2);
  for (const [x, z] of [[-0.22, 0.25], [0.22, 0.25], [-0.22, -0.25], [0.22, -0.25]]) {
    const w = add(wg, M(0x1b1d20, { roughness: 0.95, flatShading: true }), 'wheel', x, PHYS.wheelR, z); wheels.push(w);
    add(hg, alu, 'hubs', x, PHYS.wheelR, z * 1.005);
    add(mg, M(0x9aa3ad, { metalness: 0.8, roughness: 0.3 }), 'motor', x, PHYS.wheelR, Math.sign(z) * 0.16);
    add(B(0.045, 0.04, 0.004), alu, 'hubs', x, PHYS.wheelR + 0.01, Math.sign(z) * 0.205);
  }
  // Bioenno BLF-1220A battery (181 x 167 x 77 mm, est.)
  add(B(0.181, 0.167, 0.077), M(0x23272d), 'battery', -0.12, deckY + 0.0835, 0.07);
  add(B(0.03, 0.012, 0.02), M(0xc23b3b), 'battery', -0.17, deckY + 0.172, 0.07);
  // Hammond 1554 enclosure (translucent lid) with Pi 5, 2x MDD10A, buck, ADC inside
  const box = add(B(0.2, 0.075, 0.12), M(0xb8c4cc, { transparent: true, opacity: 0.35, depthWrite: false, roughness: 0.2 }), 'box', 0.13, deckY + 0.0375, 0.06);
  box.renderOrder = 2;
  add(B(0.085, 0.004, 0.056), M(0x1f7a3a), 'pi', 0.09, deckY + 0.012, 0.06);
  add(B(0.02, 0.012, 0.016), alu, 'pi', 0.09, deckY + 0.022, 0.06);
  for (const dz of [0.03, 0.09]) add(B(0.06, 0.004, 0.025), M(0xa82828), 'driver', 0.185, deckY + 0.01, dz);
  add(B(0.02, 0.004, 0.018), M(0x245ea8), 'buck', 0.2, deckY + 0.03, 0.06);
  add(B(0.028, 0.003, 0.018), M(0x1c4fa0), 'ads1115', 0.15, deckY + 0.03, 0.105);
  add(B(0.03, 0.01, 0.03), dark, 'wiring', -0.02, deckY + 0.006, 0.12);
  // sensor mast (800 mm 2020), Davis 7714 shield with SHT45, GNSS puck on top
  const mx = 0.22, mz = -0.13, top = deckY + 0.8;
  add(B(0.02, 0.8, 0.02), alu, 'rail', mx, deckY + 0.4, mz);
  add(B(0.08, 0.012, 0.012), alu, 'rail', mx - 0.04, top - 0.18, mz);
  const sx = mx - 0.085;
  for (let k = 0; k < 6; k++) add(new THREE.CylinderGeometry(0.065 - k * 0.003, 0.075 - k * 0.003, 0.012, 16), M(0xeceeee, { roughness: 0.7 }), 'shield', sx, top - 0.26 + k * 0.022, mz);
  add(B(0.018, 0.004, 0.012), M(0x2b2b8f), 'sht45', sx, top - 0.275, mz);
  add(new THREE.CylinderGeometry(0.004, 0.004, 0.05, 6), dark, 'sht45', sx, top - 0.3, mz);
  add(new THREE.CylinderGeometry(0.04, 0.042, 0.022, 18), M(0x111214), 'gnss', mx, top + 0.011, mz);
  add(new THREE.CylinderGeometry(0.006, 0.006, 0.004, 8), M(0x666a70), 'gnss', mx, top + 0.024, mz);
  // DS18B20 waterproof probe dangling from the front to leaf height
  const cab = add(new THREE.CylinderGeometry(0.004, 0.004, 0.16, 6), dark, 'ds18b20', 0.33, 0.06, -0.12); cab.rotation.z = 0.55;
  add(new THREE.CylinderGeometry(0.0035, 0.0035, 0.05, 8), alu, 'ds18b20', 0.375, -0.005 + 0.02, -0.12);
  // deploy arm: hinge at rear, two 2020 arms, cover roll, L16 actuator body + rod
  const pivot = new THREE.Group(); pivot.position.set(-0.29, 0.12, 0); grp.add(pivot);
  const armGrp = new THREE.Group(); pivot.add(armGrp);
  for (const z of [-0.2, 0.2]) add(B(0.3, 0.02, 0.02), alu, 'roll', -0.15, 0, z, armGrp);
  const rg = new THREE.CylinderGeometry(0.03, 0.03, 0.5, 14); rg.rotateX(Math.PI / 2);
  add(rg, M(0xeeeeee), 'roll', -0.3, 0, 0, armGrp);
  const wrapG = new THREE.CylinderGeometry(0.055, 0.055, 0.46, 16); wrapG.rotateX(Math.PI / 2);
  const wrap = add(wrapG, M(0xcfe9ff, { transparent: true, opacity: 0.85, roughness: 0.9 }), 'cover', -0.3, 0, 0, armGrp);
  const actBody = add(new THREE.CylinderGeometry(0.012, 0.012, 1, 10), M(0x2c3036), 'actuator', 0, 0, 0);
  const actRod = add(new THREE.CylinderGeometry(0.004, 0.004, 1, 8), alu, 'actuator', 0, 0, 0);
  const actBase = new THREE.Vector3(-0.08, deckY + 0.02, 0);
  rv = { pivot, armGrp, wrap, actBody, actRod, actBase };
  return grp;
}
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _d = new THREE.Vector3(), _up = new THREE.Vector3(0, 1, 0);
function orientSeg(mesh, a, b) {
  _d.subVectors(b, a); const L = _d.length() || 1e-6;
  mesh.position.copy(a).addScaledVector(_d, 0.5); mesh.scale.set(1, L, 1); mesh.quaternion.setFromUnitVectors(_up, _d.normalize());
}
function poseRover() {
  if (!rv) return;
  const e = rover.act / PHYS.actStroke; // 0 stowed .. 1 fully extended
  const th = 0.35 + e * 1.0; // arm angle above horizontal (rad), backwards
  rv.armGrp.rotation.z = -th;
  // attach point 0.12 m along arm
  _b.set(-0.12 * Math.cos(th), 0.12 * Math.sin(th), 0).add(rv.pivot.position);
  const L0 = 0.118; // closed body length of L16 (est.)
  _d.subVectors(_b, rv.actBase).normalize();
  _a.copy(rv.actBase).addScaledVector(_d, L0);
  orientSeg(rv.actBody, rv.actBase, _a); orientSeg(rv.actRod, _a, _b);
  rollPos.set(rover.x + rv.pivot.position.x - 0.3 * Math.cos(th), rv.pivot.position.y + 0.3 * Math.sin(th), ROVER_Z);
  rv.wrap.visible = cloth.state === 'stowed';
}

// ---------- field scene with small real-time physics ----------
const field = new THREE.Group(); field.position.x = FX; scene.add(field);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2418, roughness: 1 });
const cropMat = new THREE.MeshStandardMaterial({ color: 0x3fae4c, roughness: 0.8 });
const NCROP = 28; let crops, fieldSign, signTex, signCtx, roverGrp, wheels = [], cupGrp, clothMesh;
{
  const g = new THREE.Mesh(new THREE.BoxGeometry(9, 0.3, 8), groundMat); g.position.y = -0.17; field.add(g);
  crops = new THREE.InstancedMesh(new THREE.ConeGeometry(0.2, 0.55, 6), cropMat, NCROP);
  const m4 = new THREE.Matrix4();
  for (let i = 0; i < NCROP; i++) { m4.setPosition(-1.8 + (i % 7) * 0.6, 0.27, -1.4 + Math.floor(i / 7) * 0.75); crops.setMatrixAt(i, m4); }
  field.add(crops);
  for (let r = 0; r < 4; r++) { const ridge = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.08, 0.34), new THREE.MeshStandardMaterial({ color: 0x3a2f1e, roughness: 1 })); ridge.position.set(0, 0.02, -1.4 + r * 0.75); field.add(ridge); }
  // weather mast
  const steel = new THREE.MeshStandardMaterial({ color: 0x9aa7ba, metalness: 0.7, roughness: 0.35 });
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.07, 3.2, 8), steel); pole.position.set(3.6, 1.6, -2.8); field.add(pole);
  const arm = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.05, 0.05), steel); arm.position.set(3.6, 2.9, -2.8); field.add(arm);
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.4, 0.22), new THREE.MeshStandardMaterial({ color: 0xe8edf5 })); box.position.set(3.6, 1.9, -2.66); field.add(box);
  cupGrp = new THREE.Group(); cupGrp.position.set(3.6, 3.25, -2.8);
  for (let k = 0; k < 3; k++) { const cup = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 6), steel); cup.position.set(Math.cos(k * 2.094) * 0.28, 0, Math.sin(k * 2.094) * 0.28); cupGrp.add(cup); }
  field.add(cupGrp);
  // rover, built from the BOM parts (app/parts.js mirrors hardware/BOM.md). Scene units are metres.
  roverGrp = buildRover();
  roverGrp.position.z = ROVER_Z; field.add(roverGrp);
  signTex = document.createElement('canvas'); signTex.width = 512; signTex.height = 96; signCtx = signTex.getContext('2d');
  fieldSign = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(signTex), transparent: true, depthWrite: false }));
  fieldSign.scale.set(6.4, 1.2, 1); fieldSign.position.set(0, 4.3, 0); field.add(fieldSign);
}
function setSign(text, color) {
  signCtx.clearRect(0, 0, 512, 96); signCtx.font = '600 34px Segoe UI, sans-serif'; signCtx.textAlign = 'center'; signCtx.textBaseline = 'middle';
  signCtx.fillStyle = color; signCtx.fillText(text, 256, 48); fieldSign.material.map.needsUpdate = true;
}
// Rover: 1-D rigid body. Mass, wheel radius, motor torque/speed curve and currents come from the BOM (PHYS).
// PD drive command, limited by the 4-motor linear torque-speed curve and by tyre traction; rolling + Coulomb friction.
const rover = { x: -3.4, v: 0, m: PHYS.mass, target: -3.4, spin: 0, F: 0, amps: PHYS.idleA, soc: 0.87, act: 0, actV: 0 };
const HOME = -3.4, WORK = 0.0;
function stepRover(dt) {
  const g = 9.81, Ftrac = PHYS.mu * rover.m * g, Fstall = 4 * PHYS.stallTorque / PHYS.wheelR;
  let F = 9 * rover.m * (rover.target - rover.x) - 7 * rover.m * rover.v;
  const avail = Fstall * Math.max(0, Math.min(1, 1 - Math.sign(F) * rover.v / PHYS.vmax)); // back-EMF limit
  F = Math.max(-Math.min(avail, Ftrac), Math.min(Math.min(avail, Ftrac), F));
  const roll = -0.04 * rover.m * g * Math.sign(rover.v) - 1.5 * rover.v;
  rover.v += (F + roll) / rover.m * dt;
  if (Math.abs(rover.v) < 0.01 && Math.abs(rover.target - rover.x) < 0.02) rover.v = 0;
  rover.x += rover.v * dt; rover.spin -= rover.v * dt / PHYS.wheelR; rover.F = F;
  // actuator: constant-speed L16, extends once the rover is within 1.5 m of the work spot
  const actT = world.deploy && Math.abs(rover.x - WORK) < 1.5 ? PHYS.actStroke : 0;
  const dA = Math.max(-PHYS.actSpeed * dt, Math.min(PHYS.actSpeed * dt, actT - rover.act));
  rover.act += dA; rover.actV = dA / dt;
  const tau = Math.abs(F) * PHYS.wheelR / 4;
  const motorA = rover.v === 0 && F === 0 ? 0 : 4 * (PHYS.freeA + (PHYS.stallA - PHYS.freeA) * Math.min(1, tau / PHYS.stallTorque));
  rover.amps = PHYS.idleA + motorA + (Math.abs(rover.actV) > 1e-4 ? 0.35 : 0); // actuator ~0.35 A moving (est.)
  rover.soc = Math.max(0, rover.soc - rover.amps * dt / 3600 / PHYS.batteryAh);
}
const PHYS_RATE = 2; // physics runs at 2x wall clock so drive + deploy fits a 0.1x-speed trial
// Frost cover: mass-spring cloth (Verlet + distance constraints), drapes over a hoop frame (sphere collider) under gravity.
const CN = 11, CS = 0.5, cp = new Float32Array(CN * CN * 3), cprev = new Float32Array(CN * CN * 3);
const cloth = { state: 'stowed', timer: 0 };
const SPH = { x: 0, y: -0.9, z: -0.28, r: 2.45 };
{
  const geo = new THREE.PlaneGeometry(1, 1, CN - 1, CN - 1);
  clothMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xbfe6ff, transparent: true, opacity: 0.42, side: THREE.DoubleSide, roughness: 0.6, depthWrite: false }));
  clothMesh.frustumCulled = false; field.add(clothMesh);
  for (let i = 0; i < CN * CN; i++) cp.set([rover.x - 0.3, 0.3, ROVER_Z], i * 3);
  cprev.set(cp);
}
function stepCloth(dt) {
  const n = CN * CN;
  if (cloth.state === 'stowed' || cloth.state === 'spread') {
    const spread = cloth.state === 'spread', k = Math.min(1, (spread ? 5 : 7) * dt);
    for (let i = 0; i < n; i++) {
      const u = (i % CN) - (CN - 1) / 2, w = Math.floor(i / CN) - (CN - 1) / 2;
      const tx = spread ? u * CS : rollPos.x + u * 0.006, ty = spread ? 2.6 : rollPos.y + (i % 3) * 0.004, tz = spread ? SPH.z + w * CS : ROVER_Z + w * 0.045;
      cp[i * 3] += (tx - cp[i * 3]) * k; cp[i * 3 + 1] += (ty - cp[i * 3 + 1]) * k; cp[i * 3 + 2] += (tz - cp[i * 3 + 2]) * k;
    }
    cprev.set(cp);
    if (spread && (cloth.timer -= dt) <= 0) cloth.state = 'free';
    return;
  }
  for (let i = 0; i < n * 3; i += 3) { // Verlet with gravity + drag
    for (let a = 0; a < 3; a++) { const p = cp[i + a], v = (p - cprev[i + a]) * 0.985; cprev[i + a] = p; cp[i + a] = p + v + (a === 1 ? -9.81 * dt * dt : 0); }
  }
  for (let it = 0; it < 4; it++) {
    for (let r = 0; r < CN; r++) for (let c = 0; c < CN; c++) {
      const i = (r * CN + c) * 3;
      if (c + 1 < CN) link(i, i + 3, CS); if (r + 1 < CN) link(i, i + CN * 3, CS);
      if (c + 1 < CN && r + 1 < CN) link(i, i + CN * 3 + 3, CS * 1.4142);
    }
    for (let i = 0; i < n * 3; i += 3) { // collide with frame + ground
      const dx = cp[i] - SPH.x, dy = cp[i + 1] - SPH.y, dz = cp[i + 2] - SPH.z, d = Math.hypot(dx, dy, dz);
      if (d < SPH.r) { const s = SPH.r / (d || 1); cp[i] = SPH.x + dx * s; cp[i + 1] = SPH.y + dy * s; cp[i + 2] = SPH.z + dz * s; cprev[i] += (cp[i] - cprev[i]) * 0.5; cprev[i + 2] += (cp[i + 2] - cprev[i + 2]) * 0.5; }
      if (cp[i + 1] < 0.03) { cp[i + 1] = 0.03; cprev[i] = cp[i]; cprev[i + 2] = cp[i + 2]; }
    }
  }
}
function link(a, b, rest) {
  const dx = cp[b] - cp[a], dy = cp[b + 1] - cp[a + 1], dz = cp[b + 2] - cp[a + 2], d = Math.hypot(dx, dy, dz) || 1e-6, f = (d - rest) / d * 0.5;
  cp[a] += dx * f; cp[a + 1] += dy * f; cp[a + 2] += dz * f; cp[b] -= dx * f; cp[b + 1] -= dy * f; cp[b + 2] -= dz * f;
}
const world = { deploy: false, freeze: false, frost: 0, cold: 0, frostTarget: 0, trueT: 3.0 };
setCam(START_CAM, true);
let physAcc = 0;
const cA = new THREE.Color(), cGreen = new THREE.Color(0x3fae4c), cWhite = new THREE.Color(0xeaf6ff), cSoil = new THREE.Color(0x2a2418), cRime = new THREE.Color(0x8fa9c4);
function stepWorld(dtWall) {
  rover.target = world.deploy ? WORK : HOME;
  if (world.deploy && cloth.state === 'stowed' && Math.abs(rover.x - WORK) < 0.25 && rover.act > PHYS.actStroke * 0.95) { cloth.state = 'spread'; cloth.timer = 0.7; }
  if (!world.deploy && cloth.state !== 'stowed') cloth.state = 'stowed';
  physAcc = Math.min(0.2, physAcc + dtWall * PHYS_RATE);
  const h = 1 / 120;
  while (physAcc >= h) { stepRover(h); stepCloth(h); physAcc -= h; }
  roverGrp.position.x = rover.x; wheels.forEach((w) => w.rotation.z = rover.spin); poseRover();
  clothMesh.geometry.attributes.position.array.set(cp); clothMesh.geometry.attributes.position.needsUpdate = true; clothMesh.geometry.computeVertexNormals();
  cupGrp.rotation.y += dtWall * 3;
  const k = Math.min(1, dtWall * 2.5);
  world.frost += (world.frostTarget - world.frost) * k; world.cold += ((world.freeze ? 1 : 0) - world.cold) * k;
  cropMat.color.copy(cA.copy(cGreen).lerp(cWhite, world.frost));
  groundMat.color.copy(cA.copy(cSoil).lerp(cRime, world.cold * 0.55));
}

// ---------- playback ----------
let run = null, seg = 0, tSim = 0, playing = false, speed = 0.1, spCur = 0, stCur = 0, liveD = 0, liveH = 0, logged = false;
const trialMs = (tr) => tr.phase === 'eval' ? 250 : 400;
const actName = (a) => a === 1 ? 'DEPLOY' : a === 0 ? 'HOLD' : '?';
function segLabel(i) { const tr = run.trials[i]; return tr.phase === 'eval' ? 'eval@' + tr.ckpt + ' #' + tr.trial : 'train ' + tr.trial; }
function gotoSeg(i, keepPlaying) {
  if (!run || !run.trials.length) return;
  seg = Math.max(0, Math.min(run.trials.length - 1, i)); tSim = 0; liveD = liveH = 0; logged = false;
  spCur = run.spikes.offsets[seg]; stCur = run.stims.offsets[seg];
  const tr = run.trials[seg];
  world.deploy = false; world.frostTarget = 0; world.freeze = tr.label === 1;
  world.trueT = tr.label === 1 ? -1.2 - ((seg * 37) % 17) / 10 : 2.4 + ((seg * 53) % 31) / 10;
  setSign((tr.label === 1 ? 'FREEZE NIGHT' : tr.label === 0 ? 'MILD NIGHT' : 'NIGHT') + ' · sensing...', tr.label === 1 ? '#bfe4ff' : '#ffd9a0');
  $('scrub').value = seg; if (!keepPlaying) setPlaying(false);
  updateInfo(); drawAcc(); drawW();
}
function setPlaying(p) { playing = p && !!run; $('play').textContent = playing ? 'Pause' : 'Play'; $('play').classList.toggle('on', playing); }
function advance(dtWall) {
  if (!run || !playing) return;
  const tr = run.trials[seg], dur = trialMs(tr), t1 = tSim + dtWall * 1000 * speed;
  const sp = run.spikes, st = run.stims, e1 = sp.offsets[seg + 1], s1 = st.offsets[seg + 1];
  while (spCur < e1 && sp.t[spCur] < t1) { const ch = sp.ch[spCur]; fireSpike(ch); if (sp.t[spCur] < 50) { if (ch >= 40 && ch < 48) liveD++; else if (ch >= 48 && ch < 56) liveH++; } spCur++; }
  while (stCur < s1 && st.t[stCur] < t1) { fireStim(st.ch[stCur], st.kind[stCur]); stCur++; }
  if (tSim < 50 && t1 >= 50) decide(tr);
  tSim = t1;
  if (tSim >= dur) {
    if (seg + 1 < run.trials.length) gotoSeg(seg + 1, true); else { setPlaying(false); logEv('end of run ' + run.name, 'sys'); }
  }
}
function decide(tr) {
  world.deploy = tr.action === 1;
  world.frostTarget = tr.label === 1 && tr.action === 0 ? 1 : 0;
  const ok = tr.correct === 1;
  const what = tr.label === 1 ? (tr.action === 1 ? 'crop protected' : 'crop frosted') : (tr.action === 1 ? 'cover wasted' : 'no cover needed');
  setSign((tr.label === 1 ? 'FREEZE' : 'MILD') + ' · ' + actName(tr.action) + ' · ' + what, ok ? '#5dff8a' : '#ff5f6e');
  if (!logged) { logged = true; logEv(`trial ${tr.trial} ${tr.phase}${tr.phase === 'eval' ? '@' + tr.ckpt : ''}: ${actName(tr.action)}, ${ok ? 'correct' : 'WRONG'} (${tr.label === 1 ? 'freeze' : 'mild'} night), deploy ${tr.deploy} hold ${tr.hold}`, ok ? 'out' : 'err'); }
  updateInfo();
}
function updateInfo() {
  if (!run) return;
  const tr = run.trials[seg], decided = tSim >= 50;
  $('info').innerHTML = `<span>run</span><b>${run.name}</b><span>station night</span><b>${segLabel(seg)} <span>(${seg + 1}/${run.trials.length})</span></b>` +
    `<span>truth</span><b>${tr.label === 1 ? 'freeze night' : tr.label === 0 ? 'mild night' : 'unknown'}</b>` +
    `<span>decision</span><b style="color:${decided ? (tr.action === 1 ? 'var(--dep)' : 'var(--hold)') : 'var(--dim)'}">${decided ? actName(tr.action) : 'listening...'}</b>` +
    `<span>outcome</span><b class="${decided ? (tr.correct === 1 ? 'ok' : 'bad') : ''}">${decided ? (tr.correct === 1 ? 'correct' : 'wrong') : '...'}</b>` +
    `<span>score</span><b>${decided ? tr.score : '...'}</b>`;
}
function updateBars() {
  if (!run) return;
  const tr = run.trials[seg], d = tSim >= 50 ? tr.deploy : liveD, h = tSim >= 50 ? tr.hold : liveH, mx = Math.max(250, tr.deploy, tr.hold, d, h);
  $('barD').style.width = (100 * d / mx) + '%'; $('barH').style.width = (100 * h / mx) + '%';
  $('lblD').textContent = 'DEPLOY ' + d; $('lblH').textContent = 'HOLD ' + h;
  $('pos').textContent = `${segLabel(seg)}  t=${Math.floor(tSim)} ms`;
  $('tlFill').style.width = Math.min(100, 100 * tSim / trialMs(tr)) + '%';
}

// ---------- charts ----------
const PAL = ['#5ad1ff', '#ffb03a', '#5dff8a', '#ff5fa8', '#a47bff', '#36d6c3', '#f3e56b', '#ff8a5c', '#9aa7ba', '#c0f', '#7fa'];
function prep(cv) {
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 290, h = Number(cv.getAttribute('height'));
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + 'px'; }
  cv.style.height = h + 'px'; const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); g.font = '10px Consolas, monospace'; return { g, w, h };
}
function axes(g, w, h, y0, y1, ticks, fmt) {
  const L = 34, R = 8, T = 8, B = 16;
  g.strokeStyle = '#1c2534'; g.fillStyle = '#7b8aa3'; g.lineWidth = 1; g.textAlign = 'right'; g.textBaseline = 'middle';
  for (const v of ticks) { const y = T + (h - T - B) * (1 - (v - y0) / (y1 - y0)); g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke(); g.fillText(fmt(v), L - 4, y); }
  return { X: (f) => L + (w - L - R) * f, Y: (v) => T + (h - T - B) * (1 - (v - y0) / (y1 - y0)), L, R, T, B };
}
function line(g, pts, col, lw) { g.strokeStyle = col; g.lineWidth = lw; g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.stroke(); }
function drawAcc() {
  const { g, w, h } = prep($('cAcc')); const a = axes(g, w, h, 0, 1, [0, 0.5, 1], (v) => (v * 100) + '%');
  if (!run) return;
  const tr = run.trials.map((t, i) => ({ t, i })).filter((o) => o.t.phase === 'train' && o.t.correct >= 0);
  if (!tr.length) return;
  const n = tr.length, pts = [], cum = [];
  let s = 0;
  for (let k = 0; k < n; k++) { let c = 0, m = 0; for (let j = Math.max(0, k - 9); j <= k; j++) { c += tr[j].t.correct; m++; } s += tr[k].t.correct; pts.push([a.X(n > 1 ? k / (n - 1) : 0), a.Y(c / m)]); cum.push([a.X(n > 1 ? k / (n - 1) : 0), a.Y(s / (k + 1))]); }
  line(g, cum, '#4a5d7a', 1); line(g, pts, '#5dff8a', 1.6);
  const cur = tr.findIndex((o) => o.i >= seg); const k = cur < 0 ? n - 1 : cur;
  g.strokeStyle = '#5ad1ff'; g.lineWidth = 1; g.beginPath(); g.moveTo(pts[k][0], a.T); g.lineTo(pts[k][0], h - a.B); g.stroke();
  g.fillStyle = '#7b8aa3'; g.textAlign = 'left'; g.textBaseline = 'alphabetic'; g.fillText('trial 0', a.L, h - 3); g.textAlign = 'right'; g.fillText(String(n - 1) + '   green: window, grey: cumulative', w - a.R, h - 3);
}
function drawW() {
  const { g, w, h } = prep($('cW')); const wt = run && run.weights;
  $('wLegend').innerHTML = '';
  if (!wt || !wt.rows.length) { g.fillStyle = '#7b8aa3'; g.textAlign = 'center'; g.fillText(run ? 'no weights.csv: synapses frozen in this condition' : '', w / 2, h / 2); return; }
  const cols = wt.header.slice(1), n = wt.rows.length;
  // plot each column as change from its first value so drift is visible on one axis
  const series = cols.map((_, ci) => wt.rows.map((r) => r[ci + 1] - (/dW/i.test(cols[ci]) ? 0 : wt.rows[0][ci + 1])));
  let lo = 0, hi = 1e-9; series.forEach((s) => s.forEach((v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const a = axes(g, w, h, lo, hi, [lo + pad, 0, hi - pad], (v) => v.toFixed(3));
  const wc = ['#ffb03a', '#36d6c3', '#a47bff', '#5ad1ff'];
  series.forEach((s, ci) => { line(g, s.map((v, k) => [a.X(n > 1 ? k / (n - 1) : 0), a.Y(v)]), wc[ci % 4], 1.5); $('wLegend').innerHTML += `<span><i style="background:${wc[ci % 4]}"></i>${/dW/i.test(cols[ci]) ? '' : 'Δ '}${cols[ci].replace(/^mean_/, '')}</span>`; });
  const nTrain = run.trials.slice(0, seg + 1).filter((t) => t.phase === 'train').length, x = a.X(n > 1 ? Math.min(1, Math.max(0, nTrain - 1) / (n - 1)) : 0);
  g.strokeStyle = '#5ad1ff'; g.lineWidth = 1; g.beginPath(); g.moveTo(x, a.T); g.lineTo(x, h - a.B); g.stroke();
}
let bench = null;
function drawLC() {
  const { g, w, h } = prep($('cLC')); const a = axes(g, w, h, 0.4, 1, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], (v) => v.toFixed(1));
  if (!bench || !bench.budgets.length) { g.fillStyle = '#7b8aa3'; g.textAlign = 'center'; g.fillText('no benchmark results found', w / 2, h / 2); return; }
  const B = bench.budgets, lx = (b) => a.X(B.length > 1 ? (Math.log(b) - Math.log(B[0])) / (Math.log(B[B.length - 1]) - Math.log(B[0])) : 0.5);
  g.fillStyle = '#7b8aa3'; g.textAlign = 'center'; g.textBaseline = 'alphabetic'; B.forEach((b) => g.fillText(String(b), lx(b), h - 3));
  const sel = run ? run.condition : null; let html = '';
  const order = bench.headline.slice().sort((x, y) => (x.condition === sel) - (y.condition === sel));
  order.forEach((r) => {
    const ci = bench.headline.indexOf(r), col = PAL[ci % PAL.length], hi = r.condition === sel;
    const pts = B.filter((b) => r.stats[b]).map((b) => [lx(b), a.Y(r.stats[b].mean), r.stats[b]]);
    g.globalAlpha = hi || !sel ? 1 : 0.45; line(g, pts, col, hi ? 2.6 : 1.1);
    pts.forEach((p) => { if (p[2].sem != null) { const e = Math.abs(a.Y(p[2].mean + p[2].sem) - p[1]); g.beginPath(); g.moveTo(p[0], p[1] - e); g.lineTo(p[0], p[1] + e); g.moveTo(p[0] - 2, p[1] - e); g.lineTo(p[0] + 2, p[1] - e); g.moveTo(p[0] - 2, p[1] + e); g.lineTo(p[0] + 2, p[1] + e); g.stroke(); } g.fillStyle = col; g.beginPath(); g.arc(p[0], p[1], hi ? 2.8 : 1.6, 0, 7); g.fill(); });
    g.globalAlpha = 1;
  });
  bench.headline.forEach((r, ci) => { html += `<span style="${r.condition === sel ? 'color:#fff;font-weight:600' : ''}"><i style="background:${PAL[ci % PAL.length]}"></i>${r.condition}</span>`; });
  if (run && run.meta && typeof run.meta.final_auc === 'number' && run.meta.budget) { const x = lx(Math.max(B[0], Math.min(B[B.length - 1], run.meta.budget))), y = a.Y(Math.max(0.4, run.meta.final_auc)); g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 5, 0, 7); g.stroke(); html += '<span>○ this run</span>'; }
  $('lcLegend').innerHTML = html;
}
function benchTable(el, rows, withTag) {
  if (!rows.length) { el.innerHTML = '<tr><td>none</td></tr>'; return; }
  const B = bench.budgets, sel = run ? run.condition : null;
  el.innerHTML = '<tr><th>condition</th>' + B.map((b) => `<th>${b}</th>`).join('') + '<th>n</th></tr>' + rows.map((r) =>
    `<tr class="${r.condition === sel && !withTag ? 'sel' : ''}"><td title="${r.condition}${withTag ? ' [' + r.tag + ']' : ''}">${r.condition.replace(/^culture_/, 'c_')}${withTag ? ' ' + r.tag : ''}</td>` +
    B.map((b) => r.stats[b] ? `<td title="±${r.stats[b].sem == null ? 'n/a' : r.stats[b].sem.toFixed(3)} n=${r.stats[b].n}">${r.stats[b].mean.toFixed(3).slice(1)}${r.stats[b].sem != null ? '<span style="color:#56657d">±' + r.stats[b].sem.toFixed(3).slice(2) + '</span>' : ''}</td>` : '<td>-</td>').join('') + `<td>${r.nSeeds}</td></tr>`).join('');
}
function drawBench() { if (!bench) return; benchTable($('bench'), bench.headline, false); benchTable($('variants'), bench.variants, true); drawLC(); }

// ---------- consoles ----------
function addLine(con, text, cls) {
  const d = document.createElement('div'); d.className = cls || ''; d.textContent = text; con.appendChild(d);
  while (con.childElementCount > 400) con.removeChild(con.children[1]);
  con.scrollTop = con.scrollHeight;
}
const logEv = (t, c) => addLine($('conEv'), t, c);
const logP = (t, c) => addLine($('conLog'), t, c);

// ---------- data / UI ----------
let runs = [];
async function refreshRuns() {
  runs = await lab.listRuns();
  $('runs').innerHTML = runs.length ? '' : '<span style="color:var(--dim)">no runs in runs/ yet</span>';
  runs.forEach((r) => {
    const d = document.createElement('div'); d.className = 'run' + (run && run.name === r.name ? ' sel' : ''); d.tabIndex = 0;
    d.innerHTML = `<b>${r.condition}</b><span class="auc">${r.final_auc == null ? (r.complete ? '-' : 'running') : 'AUC ' + r.final_auc.toFixed(3)}</span><span>seed ${r.seed == null ? '?' : r.seed}</span><span>${r.nTrials} trials</span>`;
    d.onclick = () => selectRun(r.name, true); d.onkeydown = (e) => { if (e.key === 'Enter') selectRun(r.name, true); };
    $('runs').appendChild(d);
  });
  return runs;
}
async function selectRun(name, autoplay) {
  try {
    setPlaying(false);
    const r = await lab.loadRun(name);
    let nTrain = 0; r.trials.forEach((t) => { if (t.phase === 'train') nTrain++; else t.ckpt = nTrain; });
    run = r; $('scrub').max = Math.max(0, r.trials.length - 1);
    logEv(`loaded ${name}: ${r.trials.length} trials, ${r.spikes.t.length} spikes, ${r.stims.t.length} stims${r.meta && r.meta.final_auc != null ? ', final AUC ' + r.meta.final_auc.toFixed(3) : ''}`, 'sys');
    document.querySelectorAll('.run').forEach((d, i) => d.classList.toggle('sel', runs[i] && runs[i].name === name));
    gotoSeg(0); drawBench();
    if (autoplay && !REDUCED && r.trials.length) setPlaying(true);
    return true;
  } catch (e) { logEv('failed to load ' + name + ': ' + e.message, 'err'); return false; }
}
async function fillConditions() {
  const set = new Set(['culture_teach']);
  runs.forEach((r) => set.add(r.condition));
  if (bench) bench.headline.concat(bench.variants).forEach((r) => { if (/^culture/.test(r.condition)) set.add(r.condition); });
  const paint = () => { const cur = $('nrCond').value; $('nrCond').innerHTML = [...set].sort().map((c) => `<option>${c}</option>`).join(''); $('nrCond').value = set.has(cur) ? cur : (set.has('culture_teach') ? 'culture_teach' : [...set][0]); };
  paint();
  lab.conditions().then((a) => { if (a) { a.forEach((c) => set.add(c)); paint(); } }).catch(() => {});
}
const pending = new Map();
$('newrun').onsubmit = async (e) => {
  e.preventDefault(); $('msg').textContent = '';
  const o = { condition: $('nrCustom').value.trim() || $('nrCond').value, seed: $('nrSeed').value, budget: $('nrBudget').value, eval: $('nrEval').value };
  try { const { id } = await lab.startRun(o); $('nrStart').disabled = true; $('nrStart').textContent = 'Running ' + id + '...'; } catch (err) { $('msg').textContent = err.message; }
};
lab.onRunLog((p) => logP(`[${p.id}] ${p.line}`, p.stream));
lab.onRunDone(async (p) => {
  logP(`[${p.id}] exited with code ${p.code}${p.wall_s ? ' after ' + p.wall_s.toFixed(1) + 's' : ''}`, p.code === 0 ? 'sys' : 'err');
  $('nrStart').disabled = false; $('nrStart').textContent = 'Start run';
  await refreshRuns();
  if (p.code === 0 && !SELFTEST) selectRun(p.id, true);
  const w = pending.get(p.id); if (w) { pending.delete(p.id); w(p); }
});
let tail = { name: '', pos: null };
$('logSel').onchange = async () => { tail = { name: $('logSel').value, pos: null }; if (tail.name) { logP('--- tail ' + tail.name + ' ---', 'sys'); pollTail(); } };
async function pollTail() {
  if (!tail.name) return;
  try { const r = await lab.tailLog(tail.name, tail.pos); tail.pos = r.pos; r.lines.forEach((l) => logP(l, 'log')); } catch (e) { logP('tail failed: ' + e.message, 'err'); tail.name = ''; }
}
setInterval(pollTail, 2000);

$('play').onclick = () => { if (run && seg >= run.trials.length - 1 && tSim >= trialMs(run.trials[seg])) gotoSeg(0); setPlaying(!playing); };
$('prev').onclick = () => gotoSeg(seg - 1, playing);
$('next').onclick = () => gotoSeg(seg + 1, playing);
$('speed').onchange = () => { speed = Number($('speed').value) || 0.1; };
$('scrub').oninput = () => gotoSeg(Number($('scrub').value), playing);
window.addEventListener('keydown', (e) => { if (e.code === 'Space' && !/INPUT|SELECT|BUTTON/.test(document.activeElement.tagName)) { e.preventDefault(); $('play').click(); } });
window.addEventListener('resize', () => { drawAcc(); drawW(); drawLC(); });

// ---------- frame loop ----------
const col = new THREE.Color(); let last = performance.now(), frames = 0, fpsT = 0, fps = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  advance(dt);
  // neuron glow: ~80 ms decay in culture time, clamped so it stays visible at any playback speed
  const tau = Math.min(0.35, Math.max(0.07, 0.03 / speed)), dec = Math.exp(-dt / tau);
  for (let i = 0; i < N; i++) {
    const e = energy[i]; if (e <= 0) continue;
    const j = i * 3; col.setRGB(nbase[j] + (nhot[j] - nbase[j]) * e, nbase[j + 1] + (nhot[j + 1] - nbase[j + 1]) * e, nbase[j + 2] + (nhot[j + 2] - nbase[j + 2]) * e);
    neurons.setColorAt(i, col); energy[i] = e * dec < 0.02 ? 0 : e * dec;
    if (energy[i] === 0) neurons.setColorAt(i, col.setRGB(nbase[j], nbase[j + 1], nbase[j + 2]));
  }
  neurons.instanceColor.needsUpdate = true;
  const pa = pulseGeo.attributes.position.array;
  for (let i = 0; i < NP; i++) {
    const li = pulse.line[i]; if (li < 0) continue;
    pulse.k[i] += dt / 0.55;
    if (pulse.k[i] >= 1) { pulse.line[i] = -1; pa[i * 3 + 1] = 999; continue; }
    const b = bundle[li], f = pulse.k[i] * b.S, s = Math.min(b.S - 1, Math.floor(f)), u = f - s;
    for (let a = 0; a < 3; a++) pa[i * 3 + a] = b.pts[s * 3 + a] * (1 - u) + b.pts[s * 3 + 3 + a] * u;
  }
  pulseGeo.attributes.position.needsUpdate = true;
  for (const r of rings) { if (r.life <= 0) continue; r.life -= dt / 0.45; if (r.life <= 0) { r.grp.visible = false; continue; } r.mat.opacity = r.life * 0.9; const s = 1 + (1 - r.life) * 1.6; r.ring.scale.set(s, s, 1); }
  const rx0 = rover.x;
  stepWorld(dt);
  if (camMode === 'robot' && !camTween) { const dx = rover.x - rx0; camera.position.x += dx; controls.target.x += dx; }
  updateTelemetry(dt);
  if (camTween) { camTween.k = Math.min(1, camTween.k + dt / 0.9); const k = camTween.k * camTween.k * (3 - 2 * camTween.k); camera.position.lerpVectors(camTween.p0, camTween.p1, k); controls.target.lerpVectors(camTween.t0, camTween.t1, k); if (camTween.k >= 1) camTween = null; }
  controls.update();
  if (bloomOn && composer) composer.render(); else renderer.render(scene, camera);
  updateBars();
  frames++; fpsT += dt; if (fpsT > 0.5) { fps = frames / fpsT; frames = 0; fpsT = 0; $('hud').textContent = `${fps.toFixed(0)} fps · 1,024 model neurons · 64 electrodes · drag to orbit, wheel to zoom`; }
}
requestAnimationFrame(frame);

// ---------- telemetry + part picking ----------
const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const tel = { t: 0, sampleT: 0, reading: null, leaf: null };
function updateTelemetry(dt) {
  tel.sampleT -= dt; tel.t -= dt;
  if (tel.sampleT <= 0) { tel.sampleT = 1; tel.reading = world.trueT + gauss() * PHYS.shtSigma; tel.leaf = world.trueT - 0.8 + gauss() * 0.5; }
  if (tel.t > 0) return; tel.t = 0.2;
  const cells = [
    ['Speed', Math.abs(rover.v).toFixed(2), 'm/s', 'max ' + PHYS.vmax.toFixed(2)],
    ['Motor current', rover.amps.toFixed(2), 'A', '4x 37D + idle (est.)'],
    ['Battery', (rover.soc * 100).toFixed(2), '%', '20 Ah LiFePO4'],
    ['Actuator', (rover.act * 1000).toFixed(0), 'mm', 'of 100 · L16'],
    ['SHT45 air', tel.reading == null ? '-' : tel.reading.toFixed(2), '°C', 'true ' + world.trueT.toFixed(2) + ' · σ 0.1'],
    ['DS18B20 leaf', tel.leaf == null ? '-' : tel.leaf.toFixed(1), '°C', 'σ 0.5'],
    ['Cover', cloth.state.toUpperCase(), '', 'physics ' + PHYS_RATE + 'x real time'],
  ];
  $('telem').innerHTML = cells.map(([k, v, u, s]) => `<div><span>${k}</span><b>${v}<small>${u}</small></b><em>${s}</em></div>`).join('');
}
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
let pinned = null, hoverMesh = null;
function pickPart(ev) {
  const r = canvas.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(partMeshes, false).filter((h) => h.object.visible);
  if (!hits.length) return null;
  const inner = hits.find((h) => h.object.userData.part !== 'box');
  return (hits[0].object.userData.part === 'box' && inner && inner.distance - hits[0].distance < 0.15 ? inner : hits[0]).object;
}
function showTip(mesh, ev) {
  const tip = $('tip');
  if (hoverMesh && hoverMesh !== mesh && hoverMesh.material.emissive) hoverMesh.material.emissive.setHex(0x000000);
  hoverMesh = mesh;
  if (!mesh) { tip.style.display = 'none'; return; }
  if (mesh.material.emissive) mesh.material.emissive.setHex(0x2a4a00);
  const p = PARTS[mesh.userData.part];
  tip.innerHTML = `<b>${p.name}</b><span>${p.spec}</span><span class="mono">qty ${p.qty} · $${p.usd}${p.est ? ' (est.)' : ''} · ${p.kg} kg</span>`;
  const r = $('centre').getBoundingClientRect();
  tip.style.display = 'block'; tip.style.left = Math.max(0, Math.min(r.width - 280, ev.clientX - r.left + 14)) + 'px'; tip.style.top = (ev.clientY - r.top + 14) + 'px';
}
canvas.addEventListener('pointermove', (ev) => { if (!pinned) showTip(pickPart(ev), ev); });
canvas.addEventListener('click', (ev) => { const m = pickPart(ev); pinned = m; showTip(m, ev); });
// demo label for screenshots: pin the SHT45 shield label once in robot view
function demoTip() {
  const m = partMeshes.find((x) => x.userData.part === 'shield'); if (!m) return;
  const v = m.getWorldPosition(new THREE.Vector3()).project(camera), r = canvas.getBoundingClientRect();
  showTip(m, { clientX: r.left + (v.x + 1) / 2 * r.width, clientY: r.top + (1 - v.y) / 2 * r.height });
}

// ---------- tabs ----------
let fleetUI = null;
function setTab(name) {
  document.querySelectorAll('#tabs [data-tab]').forEach((b) => b.classList.toggle('on', b.dataset.tab === name));
  document.querySelectorAll('.view').forEach((v) => v.hidden = v.id !== 'view-' + name);
  if (name === 'fleet' && !fleetUI) fleetUI = initFleet($('view-fleet'), lab, { conditions: () => [...new Set(['culture_teach', ...runs.map((r) => r.condition)])] });
  if (fleetUI) fleetUI.setActive(name === 'fleet');
  if (name === 'runs') renderRunsTab();
  if (name === 'hardware') renderHardware();
  if (name === 'simulate') { resize(); drawAcc(); drawW(); drawLC(); }
}
document.querySelectorAll('#tabs [data-tab]').forEach((b) => b.onclick = () => setTab(b.dataset.tab));
function renderRunsTab() {
  $('runsTable').innerHTML = '<tr><th>run</th><th>condition</th><th>seed</th><th>trials</th><th>final AUC</th><th>status</th><th>modified</th></tr>' +
    runs.map((r) => `<tr><td>${r.name}</td><td>${r.condition}</td><td>${r.seed ?? '-'}</td><td>${r.nTrials}</td><td>${r.final_auc == null ? '-' : r.final_auc.toFixed(3)}</td><td>${r.complete ? '<span class="ok">complete</span>' : 'running'}</td><td>${new Date(r.mtime).toLocaleString()}</td></tr>`).join('');
  if (bench) { benchTable($('benchBig'), bench.headline, false); benchTable($('variantsBig'), bench.variants, true); }
}
function renderHardware() {
  let usd = 0, n = 0, kg = 0;
  $('bomTable').innerHTML = '<tr><th>part</th><th>key specs</th><th>qty</th><th>unit $</th><th>line $</th><th>kg each</th></tr>' + Object.values(PARTS).map((p) => {
    usd += p.usd * p.qty; n += p.qty; kg += p.kg * p.qty;
    return `<tr><td>${p.name}</td><td>${p.spec}</td><td>${p.qty}</td><td>${p.usd}${p.est ? '*' : ''}</td><td>${(p.usd * p.qty).toFixed(2)}</td><td>${p.kg}</td></tr>`;
  }).join('') + `<tr class="sel"><td>Total</td><td>${Object.keys(PARTS).length} line items</td><td>${n}</td><td></td><td>${usd.toFixed(0)}</td><td>${kg.toFixed(1)} kg</td></tr>`;
  $('physTable').innerHTML = [['mass', PHYS.mass + ' kg (est.)'], ['wheel radius', PHYS.wheelR + ' m'], ['stall torque / motor', PHYS.stallTorque + ' N·m'], ['no-load speed', PHYS.noLoadRpm + ' rpm → ' + PHYS.vmax.toFixed(2) + ' m/s'],
    ['stall / free current', PHYS.stallA + ' / ' + PHYS.freeA + ' A'], ['max drive force', (4 * PHYS.stallTorque / PHYS.wheelR).toFixed(0) + ' N (traction ' + (PHYS.mu * PHYS.mass * 9.81).toFixed(0) + ' N)'],
    ['battery', PHYS.batteryAh + ' Ah @ 12.8 V'], ['actuator', PHYS.actStroke * 1000 + ' mm @ ' + PHYS.actSpeed * 1000 + ' mm/s'], ['SHT45 noise', 'N(0, ' + PHYS.shtSigma + ' °C)']]
    .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
}

// ---------- boot ----------
(async function boot() {
  const report = { errors: [], loaded: {}, newRun: null };
  window.addEventListener('error', (e) => report.errors.push(String(e.message)));
  try {
    drawAcc(); drawW();
    try { bench = await lab.benchmarks(); } catch (e) { logP('benchmarks: ' + e.message, 'err'); }
    await refreshRuns(); drawBench(); fillConditions();
    try { const logs = await lab.listLogs(); logs.forEach((l) => { const o = document.createElement('option'); o.value = o.textContent = l; $('logSel').appendChild(o); }); } catch (e) {}
    logP('ready. Start a run on the left, or pick a repo log to tail.', 'sys');
    setTab(START_TAB);
    if (START_CAM === 'robot') setTimeout(demoTip, 6500);
    const first = DEFAULT_RUNS.find((n) => runs.some((r) => r.name === n)) || (runs[0] && runs[0].name);
    if (!SELFTEST) { if (first) await selectRun(first, true); return; }
    for (const r of runs) report.loaded[r.name] = (await selectRun(r.name, false)) && run.trials.length + ' trials, ' + run.spikes.t.length + ' spikes';
    const lines = []; lab.onRunLog((p) => lines.push(p.line));
    const done = new Promise((res) => pending.set('culture_reservoir_s7', res));
    await lab.startRun({ condition: 'culture_reservoir', seed: 7, budget: 10, eval: 10 });
    const p = await done; await refreshRuns();
    report.newRun = { code: p.code, lines, listed: runs.some((r) => r.name === 'culture_reservoir_s7') };
    if (first) await selectRun(first, true);
    setCam(START_CAM, true);
    await new Promise((r) => setTimeout(r, 7000));
    report.playback = { seg, tSim, fps, roverX: rover.x, cloth: cloth.state };
    setTab('fleet'); await new Promise((r) => setTimeout(r, 4000));
    report.fleet = fleetUI ? fleetUI.status() : null; setTab('hardware'); report.hardwareRows = $('bomTable').rows.length; setTab('simulate');
  } catch (e) { report.errors.push(String(e && e.stack || e)); }
  if (SELFTEST) lab.selftestReport(report);
})();
