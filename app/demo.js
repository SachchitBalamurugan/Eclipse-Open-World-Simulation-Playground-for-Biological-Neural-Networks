// Scripted, self-running Studio tour for screen recording (`electron . --demo [--demo-dry-fleet]`).
// Everything runs against the real backend; each step waits on real completion events (with timeouts).
import { gsap } from 'gsap';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function runDemo(ui, lab, opts = {}) {
  const D = ui.demo, st = D.state();
  const t0 = performance.now();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // ---- caption bar + toast ----
  const cap = document.createElement('div'); cap.id = 'demoCap'; document.body.appendChild(cap);
  const toast = document.createElement('div'); toast.id = 'demoToast'; document.body.appendChild(toast);
  gsap.set([cap, toast], { autoAlpha: 0 });
  async function say(text, hold = 0) {
    if (cap.textContent) await gsap.to(cap, { autoAlpha: 0, y: 6, duration: reduce ? 0 : 0.25 });
    cap.textContent = text;
    await gsap.fromTo(cap, { autoAlpha: 0, y: 8 }, { autoAlpha: 1, y: 0, duration: reduce ? 0 : 0.35, ease: 'power2.out' });
    if (hold) await sleep(hold);
  }
  function showToast(html, ms = 3500) {
    toast.innerHTML = html;
    gsap.killTweensOf(toast);
    gsap.timeline().fromTo(toast, { autoAlpha: 0, x: 16 }, { autoAlpha: 1, x: 0, duration: reduce ? 0 : 0.3, ease: 'power2.out' })
      .to(toast, { autoAlpha: 0, duration: reduce ? 0 : 0.3 }, '+=' + ms / 1000);
  }
  // ---- event waiting ----
  const waiters = new Set();
  lab.onStudioEvent((o) => { for (const w of [...waiters]) if (w.pred(o)) { waiters.delete(w); w.res(o); } });
  const waitEvent = (pred, ms) => new Promise((res) => {
    const w = { pred, res }; waiters.add(w);
    setTimeout(() => { if (waiters.delete(w)) res({ timeout: true }); }, ms);
  });
  const mainExit = (ms) => waitEvent((o) => o.type === 'exit' && o.slot !== 'fleet', ms);
  const waitUntil = async (fn, ms) => { const t = Date.now(); while (!fn() && Date.now() - t < ms) await sleep(200); };
  const log = (m, l) => { D.log('[demo] ' + m, l || 'sys'); console.log('[demo] ' + m); };
  const repo = decodeURIComponent(new URL('..', location.href).pathname).replace(/^\/([A-Za-z]:)/, '$1').replace(/\/$/, '');

  try {
    // 1. intro: surgical cart scene
    await ui.ready(); await ui.whenScene(60000);
    await say('Eclipse · A New ERA of training Biological Neural Networks.', 3600);
    await say('Train robots with biological neural network controllers.', 3000);
    await say('Every controller here is a simulated Izhikevich culture - no living neurons.', 3200);

    // 2. import
    D.setStep(1); D.flash(D.el('.st-left'));
    await say('1 · Import a robot: MJCF, URDF or a zip. Here: a 4-joint arm.');
    const name = 'demo_arm_' + new Date().toTimeString().slice(0, 8).replace(/:/g, '');
    await ui.importFile(repo + '/bnn_platform/examples/arm3.xml', { name });
    await waitUntil(() => st.sel && st.sel.name === name && !st.sceneLoading, 30000);
    await ui.whenScene(30000);
    D.flash(D.el(`.st-robot[data-name="${name}"]`));
    await sleep(1800);

    // 3. configure
    D.setStep(2); D.focus('bnn');
    await say('2 · Configure the BNN: 1,024 neurons on a 64-electrode MEA, teaching rule.');
    D.setRule('teach'); await sleep(500);
    await D.setField('f-neurons', 1024);
    await D.setField('f-lr', '0.02');
    await D.setField('f-seed', 7);
    await sleep(900);

    // 4. local training
    D.setStep(3); D.setPlatform('local'); D.focus('platform'); await sleep(900);
    D.focus('train'); D.setEpisodes(4);
    await say('3 · Train locally: live robot, electrode spikes and reach error per episode.');
    const doneP = waitEvent((o) => o.slot !== 'fleet' && (o.type === 'done' || o.type === 'error'), 120000);
    await D.pressTrain();
    const ex = await mainExit(120000);
    const dn = await Promise.race([doneP, sleep(10)]); const sm = (dn && dn.summary) || {};
    const f1 = (v) => (typeof v === 'number' ? v.toFixed(0) : '-');
    if (ex.timeout) log('local training timed out', 'err');
    await say(sm.eval_before && sm.eval_after ? 'Held-out error, learning off: ' + f1(sm.eval_before.median_final_err_mm) + ' → ' + f1(sm.eval_after.median_final_err_mm) + ' mm after ' + (sm.episodes || 4) + ' episodes.' : 'Held-out error before → after training, learning off.', 3800);
    const armRun = st.runId;

    // 5. fleet
    D.hideSummary();
    D.setPlatform('fleet'); D.focus('platform'); D.setVms(3); D.setDry(!!opts.dryFleet); D.setEpisodes(2);
    await say(opts.dryFleet ? '4 · Scale out: 3 Freestyle VMs, one seed each (rehearsal: dry run, no VMs).' : '4 · Scale out: 3 Freestyle VMs, one seed each - launching for real.');
    await sleep(600);
    await D.pressTrain();
    D.showTab('jobs');
    await waitEvent((o) => o.slot === 'fleet' && (o.type === 'fleet' || o.type === 'error' || o.type === 'exit'), 30000);
    const poll = setInterval(() => { if (!opts.dryFleet) D.refreshJobs(); }, 5000);
    await sleep(2500);

    // 6. past runs: replay + download
    D.showTab('runs'); await D.refreshRuns();
    await say('5 · While the VMs work: every past run is kept. Replay a recorded surgical-arm run.');
    D.flash(D.el('.st-runs'));
    await D.replayRun('legacy_surgical_teach');
    await waitEvent((o) => o.slot !== 'fleet' && o.type === 'frame', 20000);
    await sleep(9000);
    if (st.running) { await D.stop('main'); await mainExit(15000); }
    await say('Download any run as a zip: config, events, learning curve, culture state.');
    try {
      const p = await lab.studioPack(armRun || 'legacy_surgical_teach');
      D.log('packed ' + p.zip, 'ok');
      showToast(`<b>Run packed</b><span>${String(p.zip).replace(/[<>&]/g, '')}</span>`);
    } catch (e) { log('pack failed: ' + e.message, 'err'); }
    await sleep(3200);

    // 7. deploy
    D.setStep(6); D.focus('deploy');
    await say('6 · Sim-to-real: export a hardware bundle. Simulated weights do not transfer.');
    await sleep(800);
    const b = await D.exportBundle('generic');
    if (b) { D.focus('deploy'); await sleep(3000); }
    await say('Closed-loop dry run against a mock MEA: calibrate, remap, then control the arm.');
    if (b) { await D.dryRun(1); await mainExit(120000); }
    await sleep(2500);

    // 8. fleet status + teardown
    D.hideSummary(); D.showTab('jobs'); await D.refreshJobs(); clearInterval(poll);
    await say(opts.dryFleet ? 'Fleet jobs: planned only in this rehearsal.' : 'Fleet jobs: VM states polled live. Tearing down the demo VMs.');
    D.flash(D.el('.st-jobs'));
    if (!opts.dryFleet) {
      await waitUntil(() => !st.fleetRunning, 60000);
      try { const r = await lab.studioDemoTeardown(); log('teardown: ' + r.map((x) => x.vm + (x.ok ? ' deleted' : ' FAILED ' + x.msg)).join(', '), r.every((x) => x.ok) ? 'ok' : 'err'); }
      catch (e) { log('teardown: ' + e.message, 'err'); }
    }
    await sleep(3000);

    // 9. outro
    await say('Eclipse · import, train, scale, deploy. Simulated cultures - no living neurons.', 4000);
    await say('Eclipse · A New ERA of training Biological Neural Networks.', 4500);
  } catch (e) {
    log('demo error: ' + (e && e.stack || e), 'err');
    await sleep(2000);
  }
  console.log('[demo] total ' + ((performance.now() - t0) / 1000).toFixed(1) + ' s');
  await gsap.to(cap, { autoAlpha: 0, duration: 0.4 });
  await lab.demoQuit();
}

// ~30 s cut for the video (`electron . --demo-short [--demo-fleet-run=<run_id>] [--demo-fake-fleet]`):
// 0-3 s splash, 3-14 s local arm3 training mid-run, 14-27 s fleet grid on an ALREADY-RUNNING fleet run, 27-30 s outro.
// No VM is created or deleted here; --demo-fake-fleet shows synthetic cards labelled "rehearsal" (testing only).
export async function runDemoShort(ui, lab, opts = {}) {
  const D = ui.demo, st = D.state(); let t0 = performance.now();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const at = async (sec) => { const ms = sec * 1000 - (performance.now() - t0); if (ms > 0) await sleep(ms); };
  const log = (m, l) => { D.log('[demo] ' + m, l || 'sys'); console.log('[demo] ' + m); };
  const SLOGAN = 'A New <em>ERA</em> of training Biological Neural Networks';
  const splash = document.createElement('div'); splash.id = 'demoSplash';
  splash.innerHTML = `<img src="./assets/eclipse.svg" alt=""><b>ECLIPSE</b><span>${SLOGAN}</span><small>Simulated Izhikevich cultures — no living neurons</small>`;
  document.body.appendChild(splash);
  const cap = document.createElement('div'); cap.id = 'demoCap'; document.body.appendChild(cap); gsap.set(cap, { autoAlpha: 0 });
  async function say(text) {
    if (cap.textContent) await gsap.to(cap, { autoAlpha: 0, duration: reduce ? 0 : 0.2 });
    cap.textContent = text; await gsap.fromTo(cap, { autoAlpha: 0, y: 8 }, { autoAlpha: 1, y: 0, duration: reduce ? 0 : 0.35, ease: 'power2.out' });
  }
  if (!reduce) gsap.from(splash.children, { autoAlpha: 0, y: 12, duration: 0.8, stagger: 0.12, ease: 'power2.out' });
  try {
    // t=0: start local training of arm3 behind the splash so it is mid-run when revealed
    let preJobs = null;
    // prefetch VM status now: one fleet-status round trip over 4 busy VMs takes several seconds
    const pre = opts.fakeFleet ? Promise.resolve() : lab.studioFleetStatus().then((r) => { if (r && Array.isArray(r.jobs)) preJobs = r.jobs; }).catch(() => {});
    await ui.ready();
    await D.select('arm3'); await sleep(300);
    D.setRule('teach'); D.setPlatform('local'); D.setEpisodes(40);
    await D.pressTrain(); D.setStep(4);
    log('local training started');
    // hold the splash until the culture is actually driving the arm (pre-training eval takes a few s), then start the clock
    { const w = Date.now(); st.frameCount = 0; while (!(st.frameCount > 2) && Date.now() - w < 30000) await sleep(150); }
    t0 = performance.now() - 1500; log('first frames; timeline starts');
    await at(3);
    gsap.to(splash, { autoAlpha: 0, duration: reduce ? 0 : 0.7, ease: 'sine.inOut' });
    await say('Training a simulated BNN culture to control a 4-joint arm — live spikes, live error.');
    await at(14);
    await Promise.race([pre, sleep(2000)]);
    if (preJobs && preJobs.length) st.jobs = preJobs;
    D.showFleetGrid(opts.fleetRun || null, { fake: !!opts.fakeFleet });
    await say('Same robot, 4 cultures, 4 VMs — training in parallel.');
    await at(27);
    gsap.to(cap, { autoAlpha: 0, duration: reduce ? 0 : 0.3 });
    gsap.set(splash.querySelector('small'), { textContent: 'Train in simulation · scale across VMs · export the protocol' });
    await gsap.to(splash, { autoAlpha: 1, duration: reduce ? 0 : 0.6, ease: 'sine.inOut' });
    await at(30);
  } catch (e) { log('demo error: ' + (e && e.stack || e), 'err'); }
  try { if (st.running) await D.stop('main'); } catch (e) {}
  console.log('[demo] total ' + ((performance.now() - t0) / 1000).toFixed(1) + ' s');
  await lab.demoQuit();
}
