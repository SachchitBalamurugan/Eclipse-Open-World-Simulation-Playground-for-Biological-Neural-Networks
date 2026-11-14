/* Eclipse landing: one persistent WebGL scene behind the page.
   three.js 0.170 (jsDelivr importmap) + GSAP 3.13 ScrollTrigger (global, loaded before this module).
   Acts: 0 eclipse · 1 the culture · 2 train any robot · 3 fleet · 4 totality / download.
   ?scene=N jumps the camera to act N (screenshots). ?debug shows an error collector + fps. */
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const root = document.documentElement;
const params = new URLSearchParams(location.search);
const DEBUG = params.has("debug");
const SCENE_PARAM = params.has("scene") ? Math.max(0, Math.min(4, parseFloat(params.get("scene")) || 0)) : null;
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
const cores = navigator.hardwareConcurrency || 4;
const small = Math.min(innerWidth, innerHeight) < 700 || innerWidth < 820;
const LOW = small || cores <= 4 || params.has("low");
const DPR = Math.min(window.devicePixelRatio || 1, LOW ? 1.5 : 1.75);
const Q = LOW
  ? { particles: 4200, neurons: 120, pulses: 220, stars: 900, orbit: 36, bloom: false }
  : { particles: 14000, neurons: 260, pulses: 520, stars: 2200, orbit: 80, bloom: true };
if (LOW) root.classList.add("gl-low");

/* ---------- renderer / scene / camera ---------- */
const canvas = document.createElement("canvas");
canvas.className = "gl";
canvas.setAttribute("aria-hidden", "true");
document.body.prepend(canvas);

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: !Q.bloom, alpha: false, powerPreference: "high-performance" });
} catch (e) {
  canvas.remove();
  throw e; // caller (index.html) drops the .webgl class -> CSS/SVG hero fallback
}
renderer.setPixelRatio(DPR);
renderer.setSize(innerWidth, innerHeight, false);
renderer.setClearColor(0x050507, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050507, 0.018);
const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.1, 400);
camera.position.set(0, 0, 9);

const C = {
  void: new THREE.Color("#050507"),
  gold: new THREE.Color("#F2B45A"),
  hi: new THREE.Color("#FFD48A"),
  white: new THREE.Color("#FFF3D6"),
  plasma: new THREE.Color("#8FA6FF"),
  ember: new THREE.Color("#3A2A12"),
  red: new THREE.Color("#FF6A4D"),
};

/* world anchors for each act */
const E = new THREE.Vector3(0, 0, 0);          // eclipse
const CU = new THREE.Vector3(0, -9, 1);        // culture
const RB = new THREE.Vector3(11, -10.5, -5);   // robot
const FL = new THREE.Vector3(-9, -5, -17);     // fleet

/* shared uniforms */
const U = {
  uTime: { value: 0 },
  uPR: { value: DPR },
  uMorph: { value: 0 },
  uCulture: { value: 0 },
  uCorona: { value: 0 },
  uDiamond: { value: 0 },
  uRobot: { value: 0 },
  uFleet: { value: 0 },
  uTotality: { value: 0 },
  uGold: { value: C.gold }, uHi: { value: C.hi }, uWhite: { value: C.white }, uPlasma: { value: C.plasma },
};

const NOISE = /* glsl */ `
  float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
  float vnoise(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.-2.*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),u.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),u.x), u.y); }
  float fbm(vec2 p){ float v=0., a=.5; mat2 m=mat2(1.6,1.2,-1.2,1.6); for(int i=0;i<5;i++){ v+=a*vnoise(p); p=m*p; a*=.5; } return v; }
`;

/* soft glow point material (generic) */
function glowPoints({ vertex, fragColor = "vColor", additive = true, extraUniforms = {} }) {
  return new THREE.ShaderMaterial({
    uniforms: { ...U, ...extraUniforms },
    vertexShader: vertex,
    fragmentShader: /* glsl */ `
      varying vec3 vColor; varying float vAlpha;
      void main(){
        vec2 c = gl_PointCoord - .5; float d = length(c);
        float a = smoothstep(.5, .0, d); a *= a;
        float core = smoothstep(.16, .0, d);
        gl_FragColor = vec4(${fragColor} * (a + core*.9), vAlpha * a);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false,
    blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
  });
}

/* ---------- starfield ---------- */
{
  const n = Q.stars, pos = new Float32Array(n * 3), seed = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = 70 + Math.random() * 90, th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(ph) * Math.cos(th); pos[i * 3 + 1] = r * Math.cos(ph) * 0.7; pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th) - 20;
    seed[i] = Math.random();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
  const m = glowPoints({ vertex: /* glsl */ `
    attribute float aSeed; uniform float uTime, uPR; varying vec3 vColor; varying float vAlpha;
    void main(){
      vec4 mv = modelViewMatrix * vec4(position,1.); gl_Position = projectionMatrix * mv;
      float tw = .55 + .45*sin(uTime*(.4+aSeed*1.6) + aSeed*40.);
      gl_PointSize = (1.2 + pow(aSeed, 6.)*3.5) * uPR * 1.6;
      vColor = mix(vec3(.85,.84,.95), vec3(1.,.86,.62), step(.82, aSeed));
      vAlpha = (.25 + .6*pow(aSeed,3.)) * tw;
    }` });
  m.fog = false;
  scene.add(new THREE.Points(g, m));
}

/* ---------- ACT 0: eclipse ---------- */
const eclipse = new THREE.Group();
eclipse.position.copy(E);
scene.add(eclipse);

const moon = new THREE.Mesh(
  new THREE.SphereGeometry(1, 64, 48),
  new THREE.ShaderMaterial({
    uniforms: U,
    vertexShader: /* glsl */ `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: /* glsl */ `varying vec3 vN; varying vec3 vV; uniform vec3 uGold, uHi;
      void main(){
        float fres = pow(1. - max(dot(vN, vV), 0.), 4.);
        float upper = smoothstep(-.2, .9, vN.y*.6 + vN.x*.4);
        float f6 = pow(1. - max(dot(vN, vV), 0.), 7.);
        vec3 col = vec3(.0015,.0015,.002) + mix(uGold, uHi, upper) * f6 * (.18 + .45*upper);
        gl_FragColor = vec4(col, 1.);
        #include <colorspace_fragment>
      }`,
  })
);
moon.renderOrder = 1;
eclipse.add(moon);

const corona = new THREE.Mesh(
  new THREE.PlaneGeometry(10, 10),
  new THREE.ShaderMaterial({
    uniforms: U,
    vertexShader: /* glsl */ `varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: /* glsl */ `varying vec2 vP; uniform float uTime, uCorona, uTotality, uDiamond; uniform vec3 uGold, uHi, uWhite;
      ${NOISE}
      void main(){
        float r = length(vP); vec2 dir = vP / max(r, 1e-4);
        if (r < .985) discard;
        float t = uTime;
        float n1 = fbm(dir*1.7 + vec2(t*.025, -t*.018));
        float n2 = fbm(dir*5.5 + vec2(r*.9 - t*.11, n1*2.4));
        float n3 = fbm(dir*14. + vec2(r*2.2 - t*.2, n2*3.));
        float streak = pow(n2, 2.4)*2.6 + pow(n3, 3.)*1.2;
        float x = r - 1.;
        float fall = exp(-x*2.1) / (r*r);
        float rays = fall * (.32 + streak * (1.1 + .6*n1));
        float limb = exp(-x*36.) * 1.6 + exp(-x*9.)*.55;
        float halo = exp(-x*.75) * .06;
        float I = (rays + limb + halo) * (uCorona*(1.+uTotality*.35));
        I *= smoothstep(4.9, 3.2, r);
        vec3 col = mix(uGold, uHi, clamp(limb*.6 + n2*.4, 0., 1.));
        col = mix(col, uWhite, clamp(limb*.55, 0., 1.));
        col *= .85 + .4*uDiamond*exp(-length(vP - vec2(.5,.866))*2.5);
        gl_FragColor = vec4(col * I, 1.);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  })
);
eclipse.add(corona);

// diamond ring flare at ~1 o'clock on the limb
const diamond = new THREE.Mesh(
  new THREE.PlaneGeometry(6, 6),
  new THREE.ShaderMaterial({
    uniforms: U,
    vertexShader: /* glsl */ `varying vec2 vP; void main(){ vP = position.xy; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: /* glsl */ `varying vec2 vP; uniform float uDiamond; uniform vec3 uWhite, uHi;
      void main(){
        vec2 p = vP; float d = length(p);
        float core = exp(-d*d*90.)*3.;
        float glow = exp(-d*3.2)*.55;
        float sx = exp(-abs(p.y)*70.) * exp(-abs(p.x)*1.1);
        float sy = exp(-abs(p.x)*70.) * exp(-abs(p.y)*1.6);
        vec2 q = mat2(.707,-.707,.707,.707)*p;
        float sd = (exp(-abs(q.y)*120.)*exp(-abs(q.x)*3.) + exp(-abs(q.x)*120.)*exp(-abs(q.y)*3.))*.5;
        float I = (core + glow + sx*.9 + sy*.6 + sd) * uDiamond;
        vec3 col = mix(uHi, uWhite, clamp(core + sx, 0., 1.));
        gl_FragColor = vec4(col * I, 1.);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
  })
);
diamond.renderOrder = 5;
eclipse.add(diamond);
const DIAMOND_LOCAL = new THREE.Vector3(0.5, 0.866, 0).multiplyScalar(1.0);

/* ---------- ACT 1: the culture ---------- */
const culture = new THREE.Group();
culture.position.copy(CU);
scene.add(culture);

const M = Q.neurons;
const nPos = [];
{
  // clustered "ganglia" inside a flattened ellipsoid
  const centers = Array.from({ length: 7 }, () => new THREE.Vector3((Math.random() - .5) * 3.6, (Math.random() - .5) * 1.2, (Math.random() - .5) * 3.6));
  for (let i = 0; i < M; i++) {
    const c = centers[i % centers.length];
    const v = Math.random() < 0.7
      ? new THREE.Vector3(c.x + gauss() * .55, c.y + gauss() * .35, c.z + gauss() * .55)
      : new THREE.Vector3((Math.random() - .5) * 4.6, (Math.random() - .5) * 1.8, (Math.random() - .5) * 4.6);
    v.x = THREE.MathUtils.clamp(v.x, -2.5, 2.5); v.z = THREE.MathUtils.clamp(v.z, -2.5, 2.5); v.y = THREE.MathUtils.clamp(v.y, -1, 1.3);
    nPos.push(v);
  }
}
function gauss() { return (Math.random() + Math.random() + Math.random() - 1.5) / 1.5 * 1.7; }

// synapses: each neuron -> 3 near neighbours (+ occasional long-range)
const edges = []; // [from, to, length]
const outEdges = Array.from({ length: M }, () => []);
for (let i = 0; i < M; i++) {
  const d = [];
  for (let j = 0; j < M; j++) if (j !== i) d.push([nPos[i].distanceToSquared(nPos[j]), j]);
  d.sort((a, b) => a[0] - b[0]);
  const picks = [d[0][1], d[1][1], d[2][1]];
  if (Math.random() < 0.25) picks.push(d[8 + Math.floor(Math.random() * 30)][1]);
  for (const j of picks) { outEdges[i].push(edges.length); edges.push([i, j, Math.sqrt(nPos[i].distanceToSquared(nPos[j]))]); }
}
{
  const p = new Float32Array(edges.length * 6), col = new Float32Array(edges.length * 6);
  edges.forEach(([a, b], k) => {
    nPos[a].toArray(p, k * 6); nPos[b].toArray(p, k * 6 + 3);
    const c1 = C.gold, c2 = Math.random() < .3 ? C.plasma : C.hi;
    c1.toArray(col, k * 6); c2.toArray(col, k * 6 + 3);
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  var synMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
  culture.add(new THREE.LineSegments(g, synMat));
}

// neuron somas (flash on spike)
const nFlash = new Float32Array(M);
const somaGeo = new THREE.BufferGeometry();
{
  const p = new Float32Array(M * 3); nPos.forEach((v, i) => v.toArray(p, i * 3));
  somaGeo.setAttribute("position", new THREE.BufferAttribute(p, 3));
  somaGeo.setAttribute("aFlash", new THREE.BufferAttribute(nFlash, 1).setUsage(THREE.DynamicDrawUsage));
}
const somaMat = glowPoints({ vertex: /* glsl */ `
  attribute float aFlash; uniform float uPR, uCulture; uniform vec3 uGold, uWhite; varying vec3 vColor; varying float vAlpha;
  void main(){
    vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv;
    gl_PointSize = (7. + aFlash*26.) * uPR * (6. / -mv.z);
    vColor = mix(uGold*.9, uWhite*1.6, aFlash);
    vAlpha = uCulture * (.55 + aFlash);
  }` });
culture.add(new THREE.Points(somaGeo, somaMat));

// spike pulses travelling along synapses
const P = Q.pulses;
const pulse = { edge: new Int32Array(P).fill(-1), t: new Float32Array(P), speed: new Float32Array(P) };
const pulseGeo = new THREE.BufferGeometry();
const pulsePos = new Float32Array(P * 3), pulseA = new Float32Array(P);
pulseGeo.setAttribute("position", new THREE.BufferAttribute(pulsePos, 3).setUsage(THREE.DynamicDrawUsage));
pulseGeo.setAttribute("aA", new THREE.BufferAttribute(pulseA, 1).setUsage(THREE.DynamicDrawUsage));
const pulseMat = glowPoints({ vertex: /* glsl */ `
  attribute float aA; uniform float uPR, uCulture; uniform vec3 uHi, uWhite; varying vec3 vColor; varying float vAlpha;
  void main(){
    vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv;
    gl_PointSize = 9. * uPR * (6. / -mv.z) * step(.001, aA);
    vColor = mix(uHi, uWhite, .6) * 1.5; vAlpha = aA * uCulture;
  }` });
const pulsePoints = new THREE.Points(pulseGeo, pulseMat);
pulsePoints.frustumCulled = false;
culture.add(pulsePoints);

// 8x8 MEA under the culture
const MEA_Y = -1.75, MEA_S = 0.62;
const elecFlash = new Float32Array(64);
const nearestElec = new Int16Array(M);
{
  const sub = new THREE.Mesh(new THREE.PlaneGeometry(6.6, 6.6), new THREE.MeshBasicMaterial({ color: 0x0b0b10, transparent: true, opacity: 0.0, depthWrite: false }));
  sub.rotation.x = -Math.PI / 2; sub.position.y = MEA_Y - 0.01;
  culture.add(sub);
  var subMat = sub.material;

  const ep = new Float32Array(64 * 3);
  const lead = [];
  for (let i = 0; i < 64; i++) {
    const gx = (i % 8) - 3.5, gz = Math.floor(i / 8) - 3.5;
    ep[i * 3] = gx * MEA_S; ep[i * 3 + 1] = MEA_Y; ep[i * 3 + 2] = gz * MEA_S;
    // leads fan out to the nearest edge of the substrate
    const ax = Math.abs(gx) > Math.abs(gz);
    const ex = ax ? Math.sign(gx) * 3.2 : gx * 0.78, ez = ax ? gz * 0.78 : Math.sign(gz) * 3.2;
    lead.push(gx * MEA_S, MEA_Y, gz * MEA_S, ex, MEA_Y, ez);
  }
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.Float32BufferAttribute(lead, 3));
  var leadMat = new THREE.LineBasicMaterial({ color: 0x33333f, transparent: true, opacity: 0, depthWrite: false });
  culture.add(new THREE.LineSegments(lg, leadMat));
  const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(6.6, 6.6)), leadMat);
  border.rotation.x = -Math.PI / 2; border.position.y = MEA_Y; culture.add(border);

  const eg = new THREE.BufferGeometry();
  eg.setAttribute("position", new THREE.BufferAttribute(ep, 3));
  eg.setAttribute("aFlash", new THREE.BufferAttribute(elecFlash, 1).setUsage(THREE.DynamicDrawUsage));
  var elecGeo = eg;
  const em = new THREE.ShaderMaterial({
    uniforms: U,
    vertexShader: /* glsl */ `attribute float aFlash; uniform float uPR; varying float vF;
      void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv; gl_PointSize = 30.*uPR*(6./-mv.z); vF = aFlash; }`,
    fragmentShader: /* glsl */ `varying float vF; uniform float uCulture; uniform vec3 uGold, uWhite;
      void main(){
        vec2 c = abs(gl_PointCoord - .5); float d = max(c.x, c.y); float rd = length(gl_PointCoord-.5);
        float pad = smoothstep(.2, .17, d);
        float ring = smoothstep(.22, .2, d) - smoothstep(.2, .18, d);
        float glow = exp(-rd*rd*18.) * vF;
        vec3 col = vec3(.09,.085,.11)*pad + uGold*ring*.6 + mix(uGold, uWhite, vF*.6)*(pad*vF*1.4 + glow*1.2);
        gl_FragColor = vec4(col, (pad*.9 + glow) * uCulture);
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false,
  });
  culture.add(new THREE.Points(eg, em));
  nPos.forEach((v, i) => {
    const gx = Math.round(THREE.MathUtils.clamp(v.x / MEA_S + 3.5, 0, 7)), gz = Math.round(THREE.MathUtils.clamp(v.z / MEA_S + 3.5, 0, 7));
    nearestElec[i] = gz * 8 + gx;
  });
}

// Izhikevich network (CPU, ~1ms steps; a visual, not the product's simulator)
const nv = new Float32Array(M).fill(-65), nu = new Float32Array(M), nI = new Float32Array(M);
const na = new Float32Array(M), nd = new Float32Array(M), nc = new Float32Array(M);
for (let i = 0; i < M; i++) {
  const fs = Math.random() < 0.2; // fast-spiking interneurons
  na[i] = fs ? 0.1 : 0.02; nd[i] = fs ? 2 : 8 - Math.random() * 4; nc[i] = fs ? -65 : -65 + Math.random() * 15;
  nu[i] = 0.2 * nv[i];
}
let spikesThisFrame = 0, burstTimer = 2;
function stepNetwork(dt) {
  spikesThisFrame = 0;
  burstTimer -= dt;
  const burst = burstTimer < 0 ? 1 : 0;
  if (burstTimer < -0.25) burstTimer = 1.8 + Math.random() * 3;
  const steps = 3;
  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < M; i++) {
      const I = nI[i] + gauss() * 3.2 + 1.2 + burst * 7;
      nI[i] *= 0.6;
      let v = nv[i], u = nu[i];
      v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I);
      v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + I);
      u += na[i] * (0.2 * v - u);
      if (v >= 30) { v = nc[i]; u += nd[i]; spike(i); }
      nv[i] = v; nu[i] = u;
    }
  }
}
let pulseCursor = 0;
function spike(i) {
  spikesThisFrame++;
  nFlash[i] = 1;
  elecFlash[nearestElec[i]] = Math.min(1, elecFlash[nearestElec[i]] + 0.55);
  for (const e of outEdges[i]) {
    // find a free pulse slot (ring buffer, drops oldest when saturated)
    pulse.edge[pulseCursor] = e; pulse.t[pulseCursor] = 0; pulse.speed[pulseCursor] = 1.6 / Math.max(edges[e][2], 0.3);
    pulseCursor = (pulseCursor + 1) % P;
  }
}
function updatePulses(dt) {
  for (let k = 0; k < P; k++) {
    const e = pulse.edge[k];
    if (e < 0) { pulseA[k] = 0; continue; }
    let t = pulse.t[k] + dt * pulse.speed[k];
    const [a, b] = edges[e];
    if (t >= 1) { pulse.edge[k] = -1; pulseA[k] = 0; nI[b] += 9; continue; }
    pulse.t[k] = t;
    const A = nPos[a], B = nPos[b];
    pulsePos[k * 3] = A.x + (B.x - A.x) * t; pulsePos[k * 3 + 1] = A.y + (B.y - A.y) * t; pulsePos[k * 3 + 2] = A.z + (B.z - A.z) * t;
    pulseA[k] = Math.sin(t * Math.PI);
  }
  pulseGeo.attributes.position.needsUpdate = true; pulseGeo.attributes.aA.needsUpdate = true;
  const f = Math.exp(-dt * 5), fe = Math.exp(-dt * 3.2);
  for (let i = 0; i < M; i++) nFlash[i] *= f;
  for (let i = 0; i < 64; i++) elecFlash[i] *= fe;
  somaGeo.attributes.aFlash.needsUpdate = true; elecGeo.attributes.aFlash.needsUpdate = true;
}

/* corona -> culture morph particles (world space; GPU morph) */
{
  const N = Q.particles;
  const a0 = new Float32Array(N * 3), a1 = new Float32Array(N * 3), seed = new Float32Array(N), type = new Float32Array(N);
  const streamerAngles = Array.from({ length: 11 }, () => Math.random() * Math.PI * 2);
  for (let i = 0; i < N; i++) {
    // corona shell, biased into streamers
    const ang = Math.random() < 0.6 ? streamerAngles[i % 11] + gauss() * 0.09 : Math.random() * Math.PI * 2;
    const r = 1.03 + Math.pow(Math.random(), 2.2) * 2.6;
    a0[i * 3] = E.x + Math.cos(ang) * r; a0[i * 3 + 1] = E.y + Math.sin(ang) * r; a0[i * 3 + 2] = E.z + gauss() * 0.12 * r;
    // culture: soma clusters or axon dust along synapses
    let v;
    if (Math.random() < 0.5) { const n = nPos[i % M]; v = new THREE.Vector3(n.x + gauss() * .07, n.y + gauss() * .07, n.z + gauss() * .07); type[i] = 0; }
    else { const [ea, eb] = edges[i % edges.length]; const t = Math.random(); v = nPos[ea].clone().lerp(nPos[eb], t); v.x += gauss() * .03; v.y += gauss() * .03; v.z += gauss() * .03; type[i] = 1; }
    v.add(CU);
    v.toArray(a1, i * 3);
    seed[i] = Math.random();
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(a0, 3));
  g.setAttribute("aTo", new THREE.BufferAttribute(a1, 3));
  g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
  g.setAttribute("aType", new THREE.BufferAttribute(type, 1));
  const m = glowPoints({ vertex: /* glsl */ `
    attribute vec3 aTo; attribute float aSeed, aType;
    uniform float uTime, uPR, uMorph, uCorona; uniform vec3 uGold, uHi, uPlasma, uWhite; varying vec3 vColor; varying float vAlpha;
    void main(){
      float m = smoothstep(aSeed*.45, aSeed*.45 + .55, uMorph);
      float sw = uTime*(.015 + aSeed*.02);
      vec3 from = position; float cs = cos(sw), sn = sin(sw);
      from.xy = mat2(cs, -sn, sn, cs) * from.xy;
      vec3 p = mix(from, aTo, m*m*(3.-2.*m));
      float arc = sin(m*3.14159);
      p += vec3(sin(aSeed*91.+uTime*.3), cos(aSeed*47.)*.6 - .4, sin(aSeed*13.+uTime*.2)) * arc * 2.2;
      p += vec3(sin(uTime*.7+aSeed*30.), cos(uTime*.6+aSeed*20.), sin(uTime*.5+aSeed*10.)) * .025 * m;
      vec4 mv = modelViewMatrix*vec4(p,1.); gl_Position = projectionMatrix*mv;
      float sz = mix(2.2 + aSeed*2.6, 1.6 + (1.-aType)*1.6, m);
      gl_PointSize = sz * uPR * (7. / -mv.z);
      vec3 cc = mix(uGold, uHi, aSeed);
      vec3 nc = mix(mix(uHi, uWhite, .3), uPlasma, aType*.55);
      vColor = mix(cc, nc, m);
      float vis = mix(uCorona*.5, 1., m);
      vAlpha = vis * (aType > .5 ? .5 : .85) * (1. - smoothstep(1.55, 1.95, uMorph));
    }` });
  const pts = new THREE.Points(g, m);
  pts.frustumCulled = false;
  scene.add(pts);
}

/* ---------- ACT 2: robot arm ---------- */
const robot = new THREE.Group();
robot.position.copy(RB);
scene.add(robot);
const gold = new THREE.MeshStandardMaterial({ color: 0xf2b45a, metalness: 0.35, roughness: 0.42, emissive: 0x24160a, flatShading: true, transparent: true });
const dark = new THREE.MeshStandardMaterial({ color: 0x262632, metalness: 0.5, roughness: 0.45, flatShading: true, transparent: true });
const robotMats = [gold, dark];
robotMats.base = 1;
const L1 = 2.0, L2 = 1.7;
const arm = {};
{
  const base = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.25, 0.35, 8), dark); base.position.y = 0.17; robot.add(base);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(1.02, 0.05, 4, 24), gold); ring.rotation.x = Math.PI / 2; ring.position.y = 0.36; robot.add(ring);
  arm.yaw = new THREE.Group(); arm.yaw.position.y = 0.36; robot.add(arm.yaw);
  const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.55, 0.6, 6), dark); tower.position.y = 0.3; arm.yaw.add(tower);
  arm.sh = new THREE.Group(); arm.sh.position.y = 0.7; arm.yaw.add(arm.sh);
  arm.sh.add(new THREE.Mesh(new THREE.IcosahedronGeometry(0.34, 0), gold));
  const up = new THREE.Mesh(new THREE.BoxGeometry(0.34, L1, 0.34), dark); up.position.y = L1 / 2; arm.sh.add(up);
  const upStripe = new THREE.Mesh(new THREE.BoxGeometry(0.36, L1 * 0.7, 0.08), gold); upStripe.position.set(0, L1 / 2, 0.16); arm.sh.add(upStripe);
  arm.el = new THREE.Group(); arm.el.position.y = L1; arm.sh.add(arm.el);
  arm.el.add(new THREE.Mesh(new THREE.IcosahedronGeometry(0.27, 0), gold));
  const fo = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.19, L2, 6), dark); fo.position.y = L2 / 2; arm.el.add(fo);
  arm.wr = new THREE.Group(); arm.wr.position.y = L2; arm.el.add(arm.wr);
  arm.wr.add(new THREE.Mesh(new THREE.OctahedronGeometry(0.18, 0), gold));
  const f1 = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.38, 0.12), gold); f1.position.set(0.12, 0.24, 0); arm.wr.add(f1);
  const f2 = f1.clone(); f2.position.x = -0.12; arm.wr.add(f2);
  arm.tip = new THREE.Object3D(); arm.tip.position.y = 0.42; arm.wr.add(arm.tip);
  arm.q = [0, 0.3, 0.9]; arm.goal = [0, 0.3, 0.9];
  const edgeMat = new THREE.LineBasicMaterial({ color: 0xffd48a, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false });
  robotMats.push(edgeMat);
  robot.traverse((o) => { if (o.isMesh && o.material === dark) o.add(new THREE.LineSegments(new THREE.EdgesGeometry(o.geometry, 20), edgeMat)); });
}
const key = new THREE.PointLight(0xffd48a, 90, 30, 1.6); key.position.set(RB.x + 3, RB.y + 5, RB.z + 4); scene.add(key);
const fill = new THREE.PointLight(0x8fa6ff, 30, 30, 1.6); fill.position.set(RB.x - 4, RB.y + 2, RB.z - 3); scene.add(fill);
scene.add(new THREE.HemisphereLight(0x8c8a99, 0x050507, 0.35));

// target + trail
const target = new THREE.Vector3(1.8, 2.2, 1.6);
const targetMesh = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, 0], 3)),
  glowPoints({ vertex: /* glsl */ `uniform float uPR, uRobot, uTime; uniform vec3 uPlasma; varying vec3 vColor; varying float vAlpha;
    void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv; gl_PointSize = (34. + sin(uTime*4.)*6.)*uPR*(6./-mv.z); vColor = uPlasma*1.4; vAlpha = uRobot; }` }));
robot.add(targetMesh);
const targetRing = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.012, 6, 48), new THREE.MeshBasicMaterial({ color: 0x8fa6ff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
robot.add(targetRing);
const TRAIL = 110;
const trailPos = new Float32Array(TRAIL * 3), trailCol = new Float32Array(TRAIL * 3);
const trailGeo = new THREE.BufferGeometry();
trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3).setUsage(THREE.DynamicDrawUsage));
trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3).setUsage(THREE.DynamicDrawUsage));
const trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
const trail = new THREE.Line(trailGeo, trailMat); trail.frustumCulled = false; scene.add(trail);
// spike sparks emitted at the joints when the motor pool fires
const SPK = 60, spkPos = new Float32Array(SPK * 3), spkA = new Float32Array(SPK), spkVel = new Float32Array(SPK * 3);
const spkGeo = new THREE.BufferGeometry();
spkGeo.setAttribute("position", new THREE.BufferAttribute(spkPos, 3).setUsage(THREE.DynamicDrawUsage));
spkGeo.setAttribute("aA", new THREE.BufferAttribute(spkA, 1).setUsage(THREE.DynamicDrawUsage));
const spkPts = new THREE.Points(spkGeo, glowPoints({ vertex: /* glsl */ `attribute float aA; uniform float uPR, uRobot; uniform vec3 uHi, uWhite; varying vec3 vColor; varying float vAlpha;
  void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv; gl_PointSize = 10.*uPR*(6./-mv.z); vColor = mix(uHi,uWhite,.5)*1.4; vAlpha = aA*uRobot; }` }));
spkPts.frustumCulled = false; scene.add(spkPts);
let spkCursor = 0, targetTimer = 0;
const tmpV = new THREE.Vector3(), tmpW = new THREE.Vector3();

function solveIK(t) {
  const S = new THREE.Vector3(0, 0.36 + 0.7, 0);
  const d = t.clone().sub(S);
  const yaw = Math.atan2(d.x, d.z);
  const h = Math.hypot(d.x, d.z), dy = d.y;
  const D = THREE.MathUtils.clamp(Math.hypot(h, dy), 0.6, L1 + L2 + 0.4 - 0.05);
  const L2e = L2 + 0.42;
  const cosE = THREE.MathUtils.clamp((D * D - L1 * L1 - L2e * L2e) / (2 * L1 * L2e), -1, 1);
  const e = Math.acos(cosE);
  const phi = Math.atan2(h, dy);
  const s = phi - Math.atan2(L2e * Math.sin(e), L1 + L2e * Math.cos(e));
  return [yaw, s, e];
}
function updateRobot(dt, t) {
  targetTimer -= dt;
  if (targetTimer <= 0) {
    targetTimer = 3.2 + Math.random() * 1.6;
    const a = (Math.random() - 0.5) * 2.4 + 0.6, r = 1.8 + Math.random() * 1.3;
    target.set(Math.sin(a) * r, 1.0 + Math.random() * 2.4, Math.cos(a) * r);
    arm.goal = solveIK(target);
  }
  targetMesh.position.copy(target);
  targetRing.position.copy(target); targetRing.lookAt(camera.position); targetRing.scale.setScalar(1 + 0.15 * Math.sin(t * 3));
  // motor pool = spikes this frame; each spike nudges the joints toward the decoded goal
  const drive = Math.min(1, spikesThisFrame / Math.max(6, M * 0.03));
  for (let j = 0; j < 3; j++) arm.q[j] += (arm.goal[j] - arm.q[j]) * (0.012 + drive * 0.11);
  arm.yaw.rotation.y = arm.q[0]; arm.sh.rotation.x = arm.q[1]; arm.el.rotation.x = arm.q[2];
  arm.wr.rotation.x = -0.4 + 0.3 * Math.sin(t * 1.3);
  robot.updateMatrixWorld(true);
  arm.tip.getWorldPosition(tmpV);
  // trail: shift and fade
  trailPos.copyWithin(3, 0, (TRAIL - 1) * 3);
  tmpV.toArray(trailPos, 0);
  for (let i = 0; i < TRAIL; i++) {
    const f = Math.pow(1 - i / TRAIL, 2) * U.uRobot.value;
    trailCol[i * 3] = 1.0 * f; trailCol[i * 3 + 1] = 0.75 * f; trailCol[i * 3 + 2] = 0.42 * f;
  }
  trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
  // sparks at joints
  if (drive > 0.15 && U.uRobot.value > 0.01) {
    const joints = [arm.sh, arm.el, arm.wr];
    const j = joints[Math.floor(Math.random() * 3)];
    j.getWorldPosition(tmpW);
    for (let k = 0; k < 3; k++) {
      const i = spkCursor; spkCursor = (spkCursor + 1) % SPK;
      tmpW.toArray(spkPos, i * 3); spkA[i] = 1;
      spkVel[i * 3] = (Math.random() - .5) * 1.4; spkVel[i * 3 + 1] = Math.random() * 1.2; spkVel[i * 3 + 2] = (Math.random() - .5) * 1.4;
    }
  }
  for (let i = 0; i < SPK; i++) {
    if (spkA[i] <= 0) continue;
    spkA[i] = Math.max(0, spkA[i] - dt * 1.8);
    spkPos[i * 3] += spkVel[i * 3] * dt; spkPos[i * 3 + 1] += spkVel[i * 3 + 1] * dt; spkPos[i * 3 + 2] += spkVel[i * 3 + 2] * dt;
  }
  spkGeo.attributes.position.needsUpdate = true; spkGeo.attributes.aA.needsUpdate = true;
}
// a small companion culture above the robot base drives it (visual link to act 1)
{
  const n = 260, p = new Float32Array(n * 3), s = new Float32Array(n);
  for (let i = 0; i < n; i++) { const v = new THREE.Vector3(gauss(), gauss() * .5, gauss()).multiplyScalar(0.32); v.toArray(p, i * 3); s[i] = Math.random(); }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("aSeed", new THREE.BufferAttribute(s, 1));
  const orb = new THREE.Points(g, glowPoints({ vertex: /* glsl */ `attribute float aSeed; uniform float uPR, uRobot, uTime; uniform vec3 uHi, uWhite; varying vec3 vColor; varying float vAlpha;
    void main(){ float a = uTime*(.3+aSeed*.4); vec3 q = position; q.xz = mat2(cos(a),-sin(a),sin(a),cos(a))*q.xz;
      vec4 mv = modelViewMatrix*vec4(q,1.); gl_Position = projectionMatrix*mv; gl_PointSize = 3.*uPR*(6./-mv.z);
      float f = step(.985, fract(aSeed*13. + uTime*.7)); vColor = mix(uHi, uWhite, f)*(1.+f*2.); vAlpha = uRobot*(.5+f); }` }));
  orb.position.set(-2.3, 0.9, 0.6);
  robot.add(orb);
}

/* ---------- ACT 3: fleet ---------- */
const fleet = new THREE.Group();
fleet.position.copy(FL);
scene.add(fleet);
const NODES = LOW ? 5 : 7;
const nodePos = [];
const fleetLineMats = [];
for (let i = 0; i < NODES; i++) {
  const a = (i / NODES) * Math.PI * 2 + 0.3;
  const v = new THREE.Vector3(Math.cos(a) * 4.6, Math.sin(a * 2) * 0.9 + (i % 2 ? 0.5 : -0.4), Math.sin(a) * 2.6);
  nodePos.push(v);
  const node = new THREE.Group(); node.position.copy(v); fleet.add(node);
  const geo = new THREE.IcosahedronGeometry(0.5, 0);
  node.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x0b0b10, transparent: true })));
  const lm = new THREE.LineBasicMaterial({ color: 0xf2b45a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  fleetLineMats.push(lm);
  const edgesL = new THREE.LineSegments(new THREE.EdgesGeometry(geo), lm); node.add(edgesL);
  node.userData.spin = 0.2 + Math.random() * 0.3;
}
const hub = new THREE.Group(); fleet.add(hub);
{
  const geo = new THREE.IcosahedronGeometry(0.85, 1);
  hub.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x050507, transparent: true })));
  const lm = new THREE.LineBasicMaterial({ color: 0xffd48a, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  fleetLineMats.push(lm);
  hub.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), lm));
}
// node cores + orbiting cultures + data packets: one Points object, GPU animated
{
  const per = Q.orbit, packetsPer = 7;
  const curves = [];
  for (let i = 0; i < NODES; i++) {
    curves.push([new THREE.Vector3(), nodePos[i]]);
    curves.push([nodePos[i], nodePos[(i + 1) % NODES]]);
  }
  const total = NODES * (per + 1) + curves.length * packetsPer + 1;
  const A = new Float32Array(total * 3), B = new Float32Array(total * 3), Cc = new Float32Array(total * 3), K = new Float32Array(total * 4);
  let k = 0;
  const put = (a, b, c, kind, p1, p2, p3) => { a.toArray(A, k * 3); b.toArray(B, k * 3); c.toArray(Cc, k * 3); K[k * 4] = kind; K[k * 4 + 1] = p1; K[k * 4 + 2] = p2; K[k * 4 + 3] = p3; k++; };
  const zero = new THREE.Vector3();
  put(zero, zero, zero, 0, 2.2, 0, 0); // hub core
  for (let i = 0; i < NODES; i++) {
    put(nodePos[i], zero, zero, 0, 1, 0, 0); // node core
    for (let j = 0; j < per; j++) {
      const axis = new THREE.Vector3(gauss(), gauss() + 2, gauss()).normalize();
      put(nodePos[i], axis, zero, 1, 0.65 + Math.random() * 0.45, 0.6 + Math.random() * 1.2, Math.random() * 6.283);
    }
  }
  // stream lines + packets
  const lineP = [];
  for (const [a, b] of curves) {
    const mid = a.clone().lerp(b, 0.5); mid.y += 0.9 + a.distanceTo(b) * 0.12;
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    const pts = curve.getPoints(28);
    for (let s = 0; s < pts.length - 1; s++) lineP.push(pts[s].x, pts[s].y, pts[s].z, pts[s + 1].x, pts[s + 1].y, pts[s + 1].z);
    for (let j = 0; j < packetsPer; j++) put(a, mid, b, 2, Math.random(), 0.12 + Math.random() * 0.18, 0);
  }
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.Float32BufferAttribute(lineP, 3));
  const streamMat = new THREE.LineBasicMaterial({ color: 0x8fa6ff, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false });
  fleetLineMats.push(streamMat); streamMat.userData.base = 0.22;
  fleet.add(new THREE.LineSegments(lg, streamMat));

  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(A, 3));
  g.setAttribute("aB", new THREE.BufferAttribute(B, 3));
  g.setAttribute("aC", new THREE.BufferAttribute(Cc, 3));
  g.setAttribute("aK", new THREE.BufferAttribute(K, 4));
  const m = glowPoints({ vertex: /* glsl */ `
    attribute vec3 aB, aC; attribute vec4 aK; uniform float uTime, uPR, uFleet; uniform vec3 uGold, uHi, uPlasma, uWhite;
    varying vec3 vColor; varying float vAlpha;
    vec3 rot(vec3 v, vec3 k, float a){ return v*cos(a) + cross(k,v)*sin(a) + k*dot(k,v)*(1.-cos(a)); }
    void main(){
      vec3 p = position; float size = 4.; vColor = uHi; vAlpha = uFleet;
      if (aK.x < .5) { size = 26.*aK.y + sin(uTime*2. + position.x)*3.; vColor = mix(uGold, uWhite, .35)*1.3; }
      else if (aK.x < 1.5) {
        vec3 ax = normalize(aB); vec3 perp = normalize(cross(ax, vec3(.3,.1,.9)));
        p = position + rot(perp*aK.y, ax, uTime*aK.z + aK.w);
        float f = step(.97, fract(aK.w*7. + uTime*.9));
        size = 3. + f*5.; vColor = mix(uHi, uWhite, f)*(1.+f*1.5); vAlpha = uFleet*(.55+f*.45);
      } else {
        float t = fract(aK.y + uTime*aK.z);
        p = mix(mix(position, aB, t), mix(aB, aC, t), t);
        size = 7.; vColor = mix(uPlasma, uWhite, .25)*1.5; vAlpha = uFleet*sin(t*3.14159);
      }
      vec4 mv = modelViewMatrix*vec4(p,1.); gl_Position = projectionMatrix*mv;
      gl_PointSize = size*uPR*(8./-mv.z);
    }` });
  const pts = new THREE.Points(g, m); pts.frustumCulled = false; fleet.add(pts);
}

/* ---------- camera journey ---------- */
const portrait = () => innerWidth / innerHeight < 0.9;
function camKeys() {
  const pr = portrait();
  return {
    pos: [
      pr ? new THREE.Vector3(0, -1.6, 13) : new THREE.Vector3(0, 0, 9.2),
      new THREE.Vector3(CU.x + 0.4, CU.y + 4.6, CU.z + (pr ? 12.5 : 9.2)),
      new THREE.Vector3(RB.x + 5.2, RB.y + 4.2, RB.z + (pr ? 13 : 10.5)),
      new THREE.Vector3(FL.x + 1.2, FL.y + 3.4, FL.z + (pr ? 17 : 11.5)),
      pr ? new THREE.Vector3(0, -1.4, 12.5) : new THREE.Vector3(0.3, 0.2, 10.5),
    ],
    tgt: [
      pr ? new THREE.Vector3(0, -2.6, 0) : new THREE.Vector3(-2.75, 0.05, 0),
      new THREE.Vector3(CU.x + (pr ? 0 : -2.6), CU.y - 0.6, CU.z),
      new THREE.Vector3(RB.x + (pr ? 0.6 : -3.2), RB.y + 1.6, RB.z + 0.6),
      new THREE.Vector3(FL.x + (pr ? 0 : -2.6), FL.y + 0.2, FL.z),
      pr ? new THREE.Vector3(0, -2.4, 0) : new THREE.Vector3(-2.9, -1.2, 0),
    ],
  };
}
let keys = camKeys();
let posCurve = new THREE.CatmullRomCurve3(keys.pos, false, "centripetal", 0.5);

const state = { act: SCENE_PARAM ?? 0 };
const smooth = (x) => x * x * (3 - 2 * x);
const mouse = { x: 0, y: 0, sx: 0, sy: 0 };
if (!REDUCE && matchMedia("(pointer: fine)").matches) {
  addEventListener("pointermove", (e) => { mouse.x = e.clientX / innerWidth - 0.5; mouse.y = e.clientY / innerHeight - 0.5; }, { passive: true });
}
const lookAt = new THREE.Vector3();
function placeCamera(act) {
  const i = Math.min(3, Math.floor(act)), f = smooth(act - i);
  posCurve.getPoint((i + f) / 4, camera.position);
  lookAt.copy(keys.tgt[i]).lerp(keys.tgt[Math.min(4, i + 1)], f);
  mouse.sx += (mouse.x - mouse.sx) * 0.05; mouse.sy += (mouse.y - mouse.sy) * 0.05;
  camera.position.x += mouse.sx * 0.9; camera.position.y -= mouse.sy * 0.6;
  camera.lookAt(lookAt);
}

/* map act -> per-group visibility */
const bump = (a, c, w) => Math.max(0, 1 - Math.abs(a - c) / w);
function applyAct(act) {
  // 0..1 corona -> culture, 1..2 culture fades out (alpha keyed on uMorph > 1.55), 3..4 back as corona dust
  U.uMorph.value = act <= 1 ? act : act < 3 ? 1 + Math.min(1, (act - 1) * 0.9) : 0;
  U.uCulture.value = smooth(bump(act, 1, 0.85));
  U.uRobot.value = smooth(bump(act, 2, 0.85));
  U.uFleet.value = smooth(bump(act, 3, 0.85));
  U.uTotality.value = smooth(Math.max(0, act - 3));
  synMat.opacity = 0.16 * U.uCulture.value; leadMat.opacity = 0.9 * U.uCulture.value; subMat.opacity = 0.75 * U.uCulture.value;
  robotMats.forEach((m) => (m.opacity = (m.isLineBasicMaterial ? 0.5 : 1) * U.uRobot.value)); targetRing.material.opacity = U.uRobot.value;
  fleetLineMats.forEach((m) => (m.opacity = (m.userData.base ?? 0.85) * U.uFleet.value));
  fleet.traverse((o) => { if (o.isMesh) o.material.opacity = U.uFleet.value; });
  culture.visible = U.uCulture.value > 0.001;
  robot.visible = U.uRobot.value > 0.001; trail.visible = spkPts.visible = robot.visible;
  fleet.visible = U.uFleet.value > 0.001;
  // HUD
  const n = Math.round(act);
  if (n !== applyAct.last) { applyAct.last = n; window.dispatchEvent(new CustomEvent("eclipse:act", { detail: n })); }
}

/* scroll -> act (ScrollTrigger scrub smooths the proxy) */
const proxy = { y: 0 };
let anchors = [];
function measure() {
  const top = (sel) => { const el = document.querySelector(sel); return el ? el.getBoundingClientRect().top + scrollY : 0; };
  const vh = innerHeight;
  const max = Math.max(1, document.documentElement.scrollHeight - vh);
  anchors = [
    [0, 0],
    [top("#demo") - vh * 0.35, 0],
    [top("#how") - vh * 0.15, 1],
    [top("#features") - vh * 0.7, 1],
    [top("#features") - vh * 0.05, 2],
    [top("#sim-to-real") - vh * 0.5, 2],
    [top("#sim-to-real") + vh * 0.2, 3],
    [top("#download") - vh * 0.75, 3],
    [top("#download") - vh * 0.05, 4],
    [max + 1, 4],
  ].map(([y, a]) => [Math.max(0, y), a]);
  for (let i = 1; i < anchors.length; i++) if (anchors[i][0] < anchors[i - 1][0]) anchors[i][0] = anchors[i - 1][0];
}
function actFromScroll(y) {
  for (let i = 1; i < anchors.length; i++) {
    const [y0, a0] = anchors[i - 1], [y1, a1] = anchors[i];
    if (y <= y1) return y1 === y0 ? a1 : a0 + (a1 - a0) * smooth(Math.min(1, Math.max(0, (y - y0) / (y1 - y0))));
  }
  return 4;
}
const gsap = window.gsap, ScrollTrigger = window.ScrollTrigger;
measure();
if (gsap && ScrollTrigger && !REDUCE && SCENE_PARAM === null) {
  gsap.registerPlugin(ScrollTrigger);
  gsap.to(proxy, {
    y: () => ScrollTrigger.maxScroll(window), ease: "none",
    scrollTrigger: { start: 0, end: "max", scrub: 1.1, invalidateOnRefresh: true, onRefresh: measure },
  });
} else if (SCENE_PARAM === null) {
  proxy.y = scrollY;
}

/* ---------- post ---------- */
let composer = null, bloom = null;
if (Q.bloom) {
  composer = new EffectComposer(renderer);
  composer.setPixelRatio(DPR);
  composer.setSize(innerWidth, innerHeight);
  composer.addPass(new RenderPass(scene, camera));
  // three writes the clear colour in output (sRGB) encoding even into the linear composer target;
  // pre-linearise it so the void stays #050507 after OutputPass instead of lifting to grey.
  renderer.setClearColor(new THREE.Color().setRGB(0.0015, 0.0015, 0.0021, THREE.SRGBColorSpace), 1);
  bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth / 2, innerHeight / 2), 0.5, 0.2, 0.6);
  // keep the glow tight: de-weight the widest mips so the void stays black
  bloom.compositeMaterial.uniforms.bloomFactors.value = [1.0, 0.55, 0.22, 0.06, 0.0];
  composer.addPass(bloom);
  composer.addPass(new OutputPass());
}

/* ---------- intro (diamond ring flash) ---------- */
if (REDUCE) {
  U.uCorona.value = 1; U.uDiamond.value = 0.35;
} else if (gsap) {
  const tl = gsap.timeline({ delay: 0.15 });
  tl.fromTo(U.uDiamond, { value: 0 }, { value: 2.4, duration: 0.55, ease: "power2.out" }, 0)
    .fromTo(U.uCorona, { value: 0 }, { value: 1, duration: 2.6, ease: "power2.out" }, 0.25)
    .to(U.uDiamond, { value: 0.32, duration: 1.8, ease: "sine.inOut" }, 0.7)
    .add(() => gsap.to(U.uDiamond, { value: 0.5, duration: 2.6, ease: "sine.inOut", yoyo: true, repeat: -1 }));
  if (SCENE_PARAM !== null || params.has("settle")) tl.progress(1);
} else { U.uCorona.value = 1; U.uDiamond.value = 0.4; }

/* ---------- loop ---------- */
const clock = new THREE.Clock();
let running = true, raf = 0, frames = 0, fpsT = 0;
const dbg = DEBUG ? document.getElementById("__debug") : null;
function frame() {
  raf = 0;
  const dt = Math.min(clock.getDelta(), 0.05);
  U.uTime.value += dt;
  const target = SCENE_PARAM ?? actFromScroll(gsap && ScrollTrigger && SCENE_PARAM === null && !REDUCE ? proxy.y : scrollY);
  state.act = target;
  render(dt);
  if (dbg && (fpsT += dt, frames++, fpsT > 1)) { dbg.dataset.fps = String(Math.round(frames / fpsT)); frames = 0; fpsT = 0; }
  if (running && !REDUCE) raf = requestAnimationFrame(frame);
}
function render(dt) {
  applyAct(state.act);
  placeCamera(state.act);
  corona.quaternion.copy(camera.quaternion);
  diamond.position.copy(DIAMOND_LOCAL);
  diamond.quaternion.copy(camera.quaternion);
  if (U.uCulture.value > 0 || U.uRobot.value > 0) {
    stepNetwork(dt); updatePulses(dt);
  }
  if (robot.visible) updateRobot(dt, U.uTime.value);
  fleet.children.forEach((c) => { if (c.userData.spin) c.rotation.y += dt * c.userData.spin; });
  hub.rotation.y -= dt * 0.15; hub.rotation.x += dt * 0.07;
  if (composer) composer.render(); else renderer.render(scene, camera);
}
function start() { if (!raf && running) { clock.getDelta(); raf = requestAnimationFrame(frame); } }

if (REDUCE) {
  // one static frame: a settled culture state is not needed; show the eclipse at totality
  U.uTime.value = 12.0;
  stepNetwork(0.016);
  render(0.016);
} else {
  if (SCENE_PARAM !== null) {
    // warm the simulation so the frozen debug frame shows activity
    U.uTime.value = 6;
    for (let i = 0; i < 90; i++) { stepNetwork(1 / 60); updatePulses(1 / 60); if (U.uRobot.value >= 0) updateRobot(1 / 60, i / 60); applyAct(SCENE_PARAM); }
  }
  start();
  document.addEventListener("visibilitychange", () => {
    running = !document.hidden;
    if (running) start(); else if (raf) { cancelAnimationFrame(raf); raf = 0; }
  });
}

let rz = 0;
addEventListener("resize", () => {
  clearTimeout(rz);
  rz = setTimeout(() => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight, false);
    if (composer) { composer.setSize(innerWidth, innerHeight); bloom.resolution.set(innerWidth / 2, innerHeight / 2); }
    keys = camKeys(); posCurve = new THREE.CatmullRomCurve3(keys.pos, false, "centripetal", 0.5);
    measure();
    if (REDUCE) render(0);
  }, 120);
});

if (SCENE_PARAM !== null) {
  // debug: also bring the matching section into view so the DOM layer matches the act
  const sel = ["#top", "#how", "#features", "#results", "#download"][Math.round(SCENE_PARAM)];
  // (headless screenshots blank out after window scroll, so shift the content layer instead)
  const go = () => {
    const el = document.querySelector(sel), m = document.querySelector("main");
    if (!el || !m || Math.round(SCENE_PARAM) === 0) return;
    m.style.transform = "translateY(" + -(el.getBoundingClientRect().top - m.getBoundingClientRect().top - 70) + "px)";
  };
  if (document.readyState === "complete") go(); else addEventListener("load", go);
}
root.classList.add("webgl-ready");
window.__eclipseScene = { state, U, Q };
