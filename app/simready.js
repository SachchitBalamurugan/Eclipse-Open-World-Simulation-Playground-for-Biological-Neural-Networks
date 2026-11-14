// SimReady farm props in the field scene. Layout comes from assets/simready/mjcf/props.xml (MuJoCo, Z-up metres),
// meshes from app/assets/simready/*.glb (Y-up). Props are CC-BY-4.0 SimReady assets; see assets/simready/ATTRIBUTION.md.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const toBuf = (u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
// MuJoCo (x, y, z) Z-up -> three (x, z, -y) Y-up. Quaternions conjugate by the same rotation.
export const mjPos = (p, out = new THREE.Vector3()) => out.set(p[0], p[2], -p[1]);
export const mjQuat = (q, out = new THREE.Quaternion()) => out.set(q[1], q[3], -q[2], q[0]); // q = [w, x, y, z]

export function parseLoader(lab, rel) {
  return lab.readAsset(rel).then((u8) => new Promise((res, rej) => new GLTFLoader().parse(toBuf(u8), '', res, rej)));
}

function bodyQuat(b) {
  const q = b.getAttribute('quat'), e = b.getAttribute('euler');
  if (q) { const v = q.trim().split(/\s+/).map(Number); return mjQuat(v).normalize(); }
  if (e) { // MuJoCo default eulerseq "xyz" (intrinsic), radians in props.xml
    const v = e.trim().split(/\s+/).map(Number);
    const qm = new THREE.Quaternion().setFromEuler(new THREE.Euler(v[0], v[1], v[2], 'XYZ'));
    return mjQuat([qm.w, qm.x, qm.y, qm.z]).normalize();
  }
  return new THREE.Quaternion();
}

// Returns {group, counts}. Every asset type is drawn as InstancedMesh (one per GLB sub-mesh), so the 121 fence panels
// are a single draw call per sub-mesh.
export async function addSimReady(parent, lab) {
  const xml = new TextDecoder().decode(await lab.readAsset('simready/props.xml'));
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  const wb = doc.querySelector('worldbody');
  const byAsset = new Map();
  for (const b of wb.children) {
    if (b.tagName !== 'body') continue;
    const g = [...b.querySelectorAll(':scope > geom')].find((x) => x.getAttribute('mesh'));
    if (!g) continue;
    const asset = g.getAttribute('mesh').replace(/^sr_/, '').replace(/__\d+$/, '');
    const p = (b.getAttribute('pos') || '0 0 0').trim().split(/\s+/).map(Number);
    const m = new THREE.Matrix4().compose(mjPos(p), bodyQuat(b), new THREE.Vector3(1, 1, 1));
    if (!byAsset.has(asset)) byAsset.set(asset, []);
    byAsset.get(asset).push({ name: b.getAttribute('name'), m });
  }
  const group = new THREE.Group(); group.name = 'simready'; parent.add(group);
  const counts = {};
  await Promise.all([...byAsset].map(async ([asset, list]) => {
    const gltf = await parseLoader(lab, 'simready/' + asset + '.glb');
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      const mat = o.material.clone();
      if (mat.isMeshStandardMaterial) { mat.roughness = Math.max(0.55, mat.roughness); mat.metalness = Math.min(0.5, mat.metalness); }
      const im = new THREE.InstancedMesh(o.geometry, mat, list.length);
      const mm = new THREE.Matrix4();
      list.forEach((it, i) => im.setMatrixAt(i, mm.multiplyMatrices(it.m, o.matrixWorld)));
      im.instanceMatrix.needsUpdate = true; im.computeBoundingSphere(); im.name = asset;
      group.add(im);
    });
    counts[asset] = list.length;
  }));
  return { group, counts };
}
