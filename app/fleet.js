// Fleet wall: one live tile per VM (mini MEA, learning curve, load sparkline, memory, progress, log tail).
// Data comes from the main process (lab.fleetGet), which proxies fleet/server.py on :8770 or a --mock-fleet feed.
const PRICE = 0.27;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const LINE_RE = /([A-Za-z0-9_]+) s(\d+):((?:\s+\d+:[0-9.]+)+)/;

export function initFleet(root, lab, opts = {}) {
  const single = opts.single || null;
  const st = new Map(); // vm id -> per-tile state
  let data = { vms: [] }, active = true, polls = 0, confirmT = 0, lastErr = null, busy = false;
  root.innerHTML = single ? `<div class="fleet single"><div id="fWall" class="wall one"></div></div>` : `
  <div class="fleet">
    <div class="fhead">
      <div class="fstats" id="fStats"></div>
      <div class="factions">
        <span id="fMock" class="mock" hidden>MOCK DATA</span>
        <span id="fSrv" class="srv"></span>
        <button id="fCollect">Collect</button>
        <button id="fTear" class="danger">Teardown</button>
        <button id="fPopAll">Pop out all</button>
      </div>
    </div>
    <div id="fLaunchSlot" class="flaunchslot"></div>
    <div id="fWall" class="wall"></div>
    <div id="fEvents" class="fevents mono"></div>
  </div>`;
  const $ = (id) => root.querySelector('#' + id);
  let form = null;
  if (!single) {
    form = document.createElement('form'); form.className = 'flaunch';
    form.innerHTML = `<span class="lbl">Launch</span>
      <label>condition<select name="cond"></select></label>
      <label>seeds<input name="seeds" type="number" min="1" max="200" value="10"></label>
      <label>VMs<input name="n" type="number" min="1" max="10" value="10"></label>
      <button type="submit" class="primary">Launch fleet</button><span class="fmsg mono"></span>`;
    const conds = (opts.conditions ? opts.conditions() : ['culture_teach']);
    form.cond.innerHTML = conds.map((c) => `<option>${esc(c)}</option>`).join(''); form.cond.value = 'culture_teach';
    form.onsubmit = async (e) => {
      e.preventDefault();
      const body = { n_vms: Math.max(1, Math.min(10, Number(form.n.value) || 1)), condition: form.cond.value, seeds: Math.max(1, Number(form.seeds.value) || 1) };
      const est = (body.n_vms * PRICE).toFixed(2);
      msg(`launching ${body.n_vms} VMs (~$${est}/h)...`);
      try { const r = await lab.fleetPost('/api/launch', body); msg(r && r.error ? r.error : 'launch accepted' + (r && r.mock ? ' (mock)' : '')); } catch (err) { msg(err.message); }
    };
    $('fCollect').onclick = async () => { try { const r = await lab.fleetPost('/api/collect', {}); msg('collect started' + (r && r.mock ? ' (mock)' : '')); } catch (e) { msg(e.message); } };
    $('fTear').onclick = async () => {
      const b = $('fTear');
      if (Date.now() > confirmT) { confirmT = Date.now() + 4000; b.textContent = 'Click again to delete all VMs'; b.classList.add('arm'); setTimeout(() => { if (Date.now() >= confirmT) { b.textContent = 'Teardown'; b.classList.remove('arm'); } }, 4100); return; }
      confirmT = 0; b.textContent = 'Teardown'; b.classList.remove('arm');
      try { const r = await lab.fleetPost('/api/teardown', {}); msg('teardown started' + (r && r.mock ? ' (mock)' : '')); } catch (e) { msg(e.message); }
    };
    $('fPopAll').onclick = () => lab.fleetPopout(sortVms(data.vms).map((v) => v.id));
  }
  function msg(t) { if (form) form.querySelector('.fmsg').textContent = t; }
  const sortVms = (vms) => (vms || []).slice().sort((a, b) => String(a.slug || '').localeCompare(String(b.slug || ''), undefined, { numeric: true }));

  function tileState(v) {
    let s = st.get(v.id);
    if (!s) {
      s = { load: [], seen: new Set(), curve: new Map(), heat: new Float32Array(64), burst: 0, el: null, lines: [] };
      st.set(v.id, s);
    }
    return s;
  }
  function ingest(v) {
    const s = tileState(v), p = v.probe || {};
    if (Array.isArray(p.load)) { s.load.push(p.load[0]); if (s.load.length > 60) s.load.shift(); }
    for (const line of p.log || []) {
      if (s.seen.has(line)) continue;
      s.seen.add(line); s.burst += 1;
      const m = LINE_RE.exec(line);
      if (m) {
        s.curve.set(m[2], Object.fromEntries(m[3].trim().split(/\s+/).map((kv) => kv.split(':').map(Number))));
        for (let k = 0; k < 14; k++) s.heat[(Math.random() * 64) | 0] = 1;
      }
    }
    if (s.seen.size > 400) s.seen = new Set([...s.seen].slice(-200));
    s.lines = (p.log || []).slice(-6);
  }

  function tileHtml(v) {
    const p = v.probe || {}, res = p.results || 0, exp = v.expected || 0, pct = exp ? Math.min(100, 100 * res / exp) : 0;
    const mp = p.mem_total_mb ? 100 * p.mem_used_mb / p.mem_total_mb : 0, state = String(v.state || '?').toLowerCase();
    return `<article class="tile" data-id="${esc(v.id)}">
      <header><b>${esc(v.slug || v.id)}</b><i class="chip ${state === 'running' ? 'run' : ''}">${esc(state)}</i>${p.stale ? '<i class="chip warn">stale</i>' : ''}
        <span class="mono dim">shard ${v.shard != null ? esc(v.shard) + '/' + esc(v.nshards) : '-'}</span>${single ? '' : '<button class="pop" title="open this VM in its own window">Pop out</button>'}</header>
      <div class="viz"><div><div class="lbl">MEA 8x8 activity</div><canvas class="mea"></canvas></div><div><div class="lbl">AUC vs training samples</div><canvas class="lc"></canvas></div></div>
      <div class="kv cpu"><span class="lbl">CPU load</span><canvas class="spark"></canvas><span class="mono v">${p.load ? p.load[0].toFixed(2) + ' / ' + (p.nproc || '-') : '-'}</span></div>
      <div class="kv mem"><span class="lbl">Memory</span><div class="mbar"><i style="width:${mp.toFixed(1)}%"></i></div><span class="mono v">${p.mem_total_mb ? (p.mem_used_mb / 1024).toFixed(1) + ' / ' + (p.mem_total_mb / 1024).toFixed(1) + ' G' : '-'}</span></div>
      <div class="kv res"><span class="lbl">Results</span><div class="mbar res"><i style="width:${pct.toFixed(1)}%"></i></div><span class="mono v">${res}${exp ? ' / ' + exp : ''}</span></div>
      <pre class="tail mono">${(p.log || []).slice(-6).map(esc).join('\n') || (p.error ? 'probe: ' + esc(p.error) : 'waiting for log...')}</pre>
    </article>`;
  }

  function render() {
    const vms = sortVms(data.vms).filter((v) => !single || v.id === single);
    const wall = $('fWall');
    if (!single) {
      const run = vms.filter((v) => String(v.state).toLowerCase() === 'running');
      let cpus = 0, res = 0, exp = 0;
      vms.forEach((v) => { const p = v.probe || {}; if (String(v.state).toLowerCase() === 'running') cpus += p.nproc || 4; res += p.results || 0; exp += v.expected || 0; });
      const L = data.launch;
      $('fStats').innerHTML = [['VMs', vms.length], ['Running', run.length], ['vCPUs', cpus], ['$/h', '$' + (run.length * (data.price_per_vm_hour || PRICE)).toFixed(2)], ['Results', exp ? res + ' / ' + exp : res],
        ['Condition', L && vms.length ? L.condition || '-' : '-']].map(([k, v]) => `<div><span>${k}</span><b class="mono">${esc(v)}</b></div>`).join('');
      $('fMock').hidden = !data.mock;
      $('fSrv').textContent = data.mock ? 'synthetic feed' : data.serverUp ? 'fleet server :8770' + (data.error ? ' · ' + String(data.error).slice(0, 60) : '') : 'fleet server not reachable' + (lastErr ? ': ' + lastErr : '');
      $('fSrv').className = 'srv ' + (data.serverUp || data.mock ? 'ok' : 'bad');
      const ev = (data.events || []).slice(-3).map((e) => `<span class="${esc(e.level)}">${new Date(e.t * 1000).toLocaleTimeString()} ${esc(e.msg)}</span>`).join('');
      $('fEvents').innerHTML = ev || '<span class="dim">no fleet events</span>';
      // launch form: big in the empty state, compact strip otherwise
      if (!vms.length) {
        if (!wall.querySelector('.empty')) {
          wall.innerHTML = `<div class="empty"><div class="lbl">Fleet</div><h3>No VMs running</h3><p>Launch shards of the benchmark on cloud VMs ($${PRICE.toFixed(2)} per VM-hour). Each VM appears here as a live tile with its MEA activity, learning curve, CPU, memory and log tail.</p><div class="slot"></div></div>`;
          wall.querySelector('.slot').appendChild(form);
        }
        for (const id of [...st.keys()]) st.delete(id);
        return;
      }
      if (form.parentElement !== $('fLaunchSlot')) { $('fLaunchSlot').appendChild(form); wall.innerHTML = ''; }
    }
    const ids = new Set(vms.map((v) => v.id));
    for (const el of [...wall.querySelectorAll('.tile')]) if (!ids.has(el.dataset.id)) el.remove();
    vms.forEach((v, i) => {
      const s = tileState(v), tmp = document.createElement('div'); tmp.innerHTML = tileHtml(v);
      const fresh = tmp.firstElementChild;
      if (s.el && s.el.isConnected) {
        // replace text parts but keep the canvases (they animate continuously)
        for (const sel of ['header', '.kv.cpu .v', '.kv.mem', '.kv.res', '.tail']) {
          const a = s.el.querySelector(sel), b = fresh.querySelector(sel); if (a && b) a.replaceWith(b);
        }
      } else { s.el = fresh; wall.appendChild(fresh); }
      if (wall.children[i] !== s.el) wall.insertBefore(s.el, wall.children[i] || null);
      const pop = s.el.querySelector('.pop'); if (pop) pop.onclick = () => lab.fleetPopout([v.id]);
      s.v = v; drawCurve(s); drawSpark(s);
    });
    if (single && !vms.length) wall.innerHTML = `<div class="empty"><h3>VM ${esc(single)} is not in the fleet any more</h3></div>`;
  }

  function prep(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return null;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h); return { g, w, h };
  }
  function drawCurve(s) {
    const c = prep(s.el.querySelector('.lc')); if (!c) return; const { g, w, h } = c;
    const L = 26, R = 6, T = 6, B = 14, y0 = 0.4, y1 = 1;
    g.font = '9px "JetBrains Mono", Consolas, monospace'; g.strokeStyle = '#262626'; g.fillStyle = '#7a7a7a'; g.lineWidth = 1; g.textAlign = 'right'; g.textBaseline = 'middle';
    const Y = (v) => T + (h - T - B) * (1 - (v - y0) / (y1 - y0));
    for (const v of [0.5, 0.75, 1]) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - R, Y(v)); g.stroke(); g.fillText(v.toFixed(2), L - 3, Y(v)); }
    const per = {}; for (const auc of s.curve.values()) for (const [b, v] of Object.entries(auc)) (per[b] = per[b] || []).push(v);
    const Bs = Object.keys(per).map(Number).sort((a, b) => a - b);
    if (!Bs.length) { g.textAlign = 'center'; g.fillText('no culture result lines yet', (L + w) / 2, h / 2); return; }
    const lb0 = Math.log(Bs[0]), lb1 = Math.log(Bs[Bs.length - 1]);
    const X = (b) => L + (w - L - R) * (Bs.length > 1 ? (Math.log(b) - lb0) / (lb1 - lb0) : 0.5);
    g.textAlign = 'center'; g.textBaseline = 'alphabetic'; Bs.forEach((b) => g.fillText(String(b), X(b), h - 2));
    // individual seeds faint, mean bright
    g.strokeStyle = 'rgba(118,185,0,0.22)';
    for (const auc of s.curve.values()) { g.beginPath(); Bs.forEach((b, i) => { const v = auc[b]; if (v == null) return; i ? g.lineTo(X(b), Y(v)) : g.moveTo(X(b), Y(v)); }); g.stroke(); }
    g.strokeStyle = '#76b900'; g.lineWidth = 2; g.beginPath();
    Bs.forEach((b, i) => { const a = per[b], m = a.reduce((x, y) => x + y, 0) / a.length; i ? g.lineTo(X(b), Y(m)) : g.moveTo(X(b), Y(m)); }); g.stroke();
    g.fillStyle = '#d0d0d0'; g.textAlign = 'right'; g.textBaseline = 'top'; g.fillText(s.curve.size + ' seeds', w - R, T);
  }
  function drawSpark(s) {
    const c = prep(s.el.querySelector('.spark')); if (!c) return; const { g, w, h } = c;
    const n = (s.v && s.v.probe && s.v.probe.nproc) || 4, a = s.load; if (a.length < 2) return;
    g.fillStyle = 'rgba(118,185,0,0.12)'; g.strokeStyle = '#76b900'; g.lineWidth = 1.2; g.beginPath(); g.moveTo(0, h);
    a.forEach((v, i) => g.lineTo(w * i / (a.length - 1), h - 1 - (h - 2) * Math.min(1, v / (n * 1.25))));
    g.lineTo(w, h); g.closePath(); g.fill(); g.beginPath();
    a.forEach((v, i) => { const x = w * i / (a.length - 1), y = h - 1 - (h - 2) * Math.min(1, v / (n * 1.25)); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  }
  // MEA animation: electrodes fire at a rate set by CPU load, plus bursts for each new log line
  let last = performance.now();
  function anim(now) {
    requestAnimationFrame(anim);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (!active) return;
    for (const s of st.values()) {
      if (!s.el || !s.el.isConnected) continue;
      const p = (s.v && s.v.probe) || {}, util = p.load && p.nproc ? Math.min(1.5, p.load[0] / p.nproc) : 0.05;
      const rate = 6 + 60 * util + 120 * Math.min(3, s.burst); s.burst = Math.max(0, s.burst - dt * 0.8);
      const fires = rate * dt; for (let k = 0; k < fires + (Math.random() < fires % 1 ? 1 : 0) - (fires % 1); k++) s.heat[(Math.random() * 64) | 0] = 1;
      const c = prep(s.el.querySelector('.mea')); if (!c) continue; const { g, w, h } = c;
      const sz = Math.min(w, h), pad = sz * 0.06, step = (sz - 2 * pad) / 8, ox = (w - sz) / 2 + pad, oy = (h - sz) / 2 + pad;
      g.fillStyle = '#050505'; g.fillRect((w - sz) / 2, (h - sz) / 2, sz, sz); g.strokeStyle = '#262626'; g.strokeRect((w - sz) / 2 + 0.5, (h - sz) / 2 + 0.5, sz - 1, sz - 1);
      const dec = Math.exp(-dt / 0.18);
      for (let i = 0; i < 64; i++) {
        const e = s.heat[i], x = ox + (i % 8 + 0.5) * step, y = oy + ((i >> 3) + 0.5) * step;
        g.fillStyle = '#1c1c1c'; g.beginPath(); g.arc(x, y, step * 0.2, 0, 7); g.fill();
        if (e > 0.02) { g.fillStyle = `rgba(${(118 + 137 * e) | 0},${(185 + 70 * e) | 0},${(0 + 120 * e) | 0},${Math.min(1, e * 1.2)})`; g.beginPath(); g.arc(x, y, step * (0.2 + 0.18 * e), 0, 7); g.fill(); }
        s.heat[i] = e * dec;
      }
    }
  }
  requestAnimationFrame(anim);

  async function poll() {
    if (!active) return;
    try { data = await lab.fleetGet(); lastErr = null; } catch (e) { lastErr = e.message; data = { vms: [], serverUp: false }; }
    polls++;
    (data.vms || []).forEach(ingest);
    render();
  }
  poll(); const timer = setInterval(poll, 3000);
  return {
    setActive(a) { active = a; if (a) poll(); },
    status: () => ({ polls, vms: (data.vms || []).length, mock: !!data.mock, serverUp: !!data.serverUp, tiles: root.querySelectorAll('.tile').length, empty: !!root.querySelector('.empty') }),
    stop: () => clearInterval(timer),
  };
}
