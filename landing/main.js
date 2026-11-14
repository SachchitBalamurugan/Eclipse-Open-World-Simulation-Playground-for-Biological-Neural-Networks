/* Eclipse landing page. GSAP 3.13 + ScrollTrigger (cdnjs). No build step. */
(function () {
  "use strict";

  /* ---------- Download manifest (works without GSAP) ---------- */
  const REPO = "https://github.com/SachchitBalamurugan/neurorover";

  function fmtBytes(n) {
    if (typeof n !== "number" || !isFinite(n)) return String(n);
    const u = ["B", "KB", "MB", "GB"]; let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return n.toFixed(i >= 2 ? 1 : 0) + " " + u[i];
  }

  async function loadManifest() {
    const $ = (id) => document.getElementById(id);
    const status = $("dl-status"), btn = $("dl-btn"), meta = $("dl-meta"), fallback = $("dl-fallback");
    try {
      const res = await fetch("downloads/manifest.json", { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const m = await res.json();
      const file = m.file || m.filename || m.name || m.zip;
      if (!file) throw new Error("manifest has no file");
      const size = m.size_bytes ?? m.size ?? m.bytes;
      const sha = m.sha256 || m.sha || "";
      let req = m.requirements || m.requires || "Windows 10/11 x64";
      if (Array.isArray(req)) req = req.join(" · ");
      else if (req && typeof req === "object") req = Object.entries(req).map(([k, v]) => k + ": " + v).join(" · ");

      $("dl-version").textContent = m.version ? "v" + String(m.version).replace(/^v/, "") : "–";
      $("dl-size").textContent = size != null ? (typeof size === "number" ? fmtBytes(size) : size) : "–";
      $("dl-sha").textContent = sha || "–";
      $("dl-req").textContent = req;
      meta.hidden = false;
      btn.href = (typeof m.url === "string" && /^https:\/\//.test(m.url)) ? m.url : "downloads/" + encodeURIComponent(file).replace(/%2F/g, "/");
      btn.setAttribute("download", "");
      btn.textContent = "Download for Windows" + (m.version ? " · v" + String(m.version).replace(/^v/, "") : "");
      const built = m.built_at || m.built || m.date;
      status.textContent = file + (built ? " · built " + String(built).slice(0, 10) : "");
    } catch (e) {
      status.textContent = "Build in progress";
      btn.href = REPO;
      btn.textContent = "View on GitHub";
      btn.classList.replace("btn--primary", "btn--ghost");
      fallback.hidden = false;
    }
  }
  loadManifest();

  /* ---------- Copy run-from-source commands ---------- */
  document.querySelectorAll("[data-copy]").forEach((b) => {
    b.addEventListener("click", async () => {
      const code = b.previousElementSibling.innerText
        .split("\n").filter((l) => l.trim() && !l.trim().startsWith("#")).join("\n");
      try { await navigator.clipboard.writeText(code); b.textContent = "Copied"; }
      catch (_) { b.textContent = "Select and copy manually"; }
      setTimeout(() => (b.textContent = "Copy commands"), 1800);
    });
  });

  /* ---------- Animation ---------- */
  if (!window.gsap) return; // page is fully readable without GSAP
  gsap.registerPlugin(ScrollTrigger);
  const SETTLE = /[?&](settle|scene)/.test(location.search);
  const WEBGL = document.documentElement.classList.contains("webgl");

  /* Split the hero headline into words (real text stays in the DOM; no paid plugins). */
  const title = document.querySelector(".hero__title");
  if (title) {
    title.setAttribute("aria-label", title.textContent.replace(/\s+/g, " ").trim());
    const parts = [];
    title.childNodes.forEach((n) => {
      if (n.nodeType === 3) n.textContent.split(/(\s+)/).forEach((t) => t && parts.push(t));
      else parts.push(n);
    });
    title.textContent = "";
    parts.forEach((p) => {
      if (typeof p === "string" && /^\s+$/.test(p)) { title.appendChild(document.createTextNode(" ")); return; }
      const w = document.createElement("span"); w.className = "w"; w.setAttribute("aria-hidden", "true");
      const inner = document.createElement("span");
      if (typeof p === "string") inner.textContent = p; else inner.appendChild(p);
      w.appendChild(inner); title.appendChild(w);
    });
  }

  /* Act HUD (driven by scene.js) */
  const ACTS = ["The eclipse", "The culture", "Train any robot", "Fleet", "Totality"];
  const hudN = document.querySelector(".act-hud__n"), hudL = document.querySelector(".act-hud__label"), hudBar = document.querySelector(".act-hud__bar i");
  window.addEventListener("eclipse:act", (e) => {
    const i = e.detail;
    if (hudN) hudN.textContent = String(i + 1).padStart(2, "0");
    if (hudL) hudL.textContent = ACTS[i] || "";
    if (hudBar) hudBar.style.transform = "scaleX(" + (0.2 + 0.2 * i) + ")";
  });
  gsap.defaults({ ease: "power2.out", duration: 0.8 });

  const phaseLabels = ["First contact", "Partial", "Partial, deeper", "Totality", "Diamond ring", "Third contact"];
  const num = document.querySelector(".how__num");
  const label = document.querySelector(".how__label");
  const steps = gsap.utils.toArray(".step");
  function setPhase(i) {
    steps.forEach((s, j) => s.classList.toggle("is-active", j === i));
    if (num) num.textContent = String(i + 1).padStart(2, "0");
    if (label) label.textContent = phaseLabels[i];
  }

  // Phase-disc positions (moon offset in the 400x400 "how" disc) for the 6 steps.
  const PH = [
    { x: -186, y: 26, ring: 0, corona: 0, diamond: 0 },
    { x: -112, y: 16, ring: 0, corona: 0, diamond: 0 },
    { x: -52, y: 8, ring: 0, corona: 0, diamond: 0 },
    { x: 0, y: 0, ring: 1, corona: 0.9, diamond: 0 },
    { x: 10, y: 6, ring: 0.8, corona: 0.5, diamond: 1 },
    { x: 170, y: -20, ring: 0, corona: 0, diamond: 0 },
  ];

  const mm = gsap.matchMedia();

  mm.add(
    { motion: "(prefers-reduced-motion: no-preference)", reduce: "(prefers-reduced-motion: reduce)" },
    (ctx) => {
      const { reduce } = ctx.conditions;

      /* Hero eclipse */
      if (WEBGL || reduce) {
        if (!WEBGL) {
        // Static totality with a visible diamond-ring point; no movement.
        gsap.set(".e-moon, .e-rim", { x: 0, y: 0 });
        gsap.set(".e-halo", { opacity: 0.12 });
        gsap.set(".e-diamond", { scale: 0.45, opacity: 0.9 });
        gsap.set(".e-corona", { opacity: 1, scale: 1 });
        }
        if (WEBGL && !reduce && !SETTLE) {
          const htl0 = gsap.timeline({ delay: 0.35, defaults: { ease: "power3.out" } });
          htl0.from(".hero__title .w > span", { yPercent: 110, rotation: 4, duration: 1.1, stagger: 0.07 }, 0)
              .from(".eyebrow", { autoAlpha: 0, y: 12, duration: 0.8 }, 0.1)
              .from(".hero__sub, .hero__ctas, .hero__note", { autoAlpha: 0, y: 22, duration: 0.9, stagger: 0.12 }, 0.6);
        }
      } else {
        const tl = gsap.timeline({ defaults: { ease: "sine.inOut" }, delay: 0.2 });
        tl.set(".e-moon, .e-rim", { x: -330, y: 230 })
          .set(".e-corona", { opacity: 0.18, scale: 0.86 })
          .set(".e-diamond", { opacity: 0, scale: 0.2 })
          .from(".hero__copy > *", { y: 24, autoAlpha: 0, duration: 0.9, stagger: 0.12, ease: "power2.out" }, 0)
          .to(".e-moon, .e-rim", { x: -7, y: 5, duration: 3.2, ease: "power2.inOut" }, 0.2)
          .to(".e-halo", { opacity: 0.12, duration: 3.2 }, 0.2)
          .addLabel("diamond", ">-0.25")
          .to(".e-diamond", { opacity: 1, scale: 1.15, duration: 0.45, ease: "power2.out" }, "diamond")
          .to(".e-diamond", { rotation: 45, duration: 1.4, ease: "sine.inOut" }, "diamond")
          .to(".e-moon, .e-rim", { x: 0, y: 0, duration: 0.9 }, "diamond+=0.35")
          .addLabel("totality", "diamond+=0.7")
          .to(".e-diamond", { opacity: 0, scale: 0.4, duration: 0.7, ease: "power2.in" }, "totality-=0.2")
          .to(".e-corona", { opacity: 1, scale: 1.08, duration: 1.2, ease: "power2.out" }, "totality-=0.3")
          .to(".e-corona", { scale: 1, duration: 1.4 }, ">")
          .add(() => {
            // The corona breathes during totality.
            gsap.to(".e-corona", { scale: 1.035, opacity: 0.86, duration: 3.2, ease: "sine.inOut", yoyo: true, repeat: -1 });
            gsap.to(".e-rays", { rotation: 8, duration: 24, ease: "sine.inOut", yoyo: true, repeat: -1 });
            gsap.to(".e-ring", { opacity: 0.65, duration: 2.1, ease: "sine.inOut", yoyo: true, repeat: -1 });
          });
        // ?settle jumps the intro to totality (used for headless screenshots).
        if (SETTLE) { tl.delay(0).pause(); tl.progress(1); }
      }

      /* How it works: moon progresses across the sun as you scroll */
      gsap.set(".h-moon", { x: PH[0].x, y: PH[0].y });
      if (reduce) {
        steps.forEach((s, i) => {
          ScrollTrigger.create({
            trigger: s, start: "top 60%", end: "bottom 60%",
            onToggle: (self) => {
              if (!self.isActive) return;
              setPhase(i);
              const p = PH[i];
              gsap.set(".h-moon", { x: p.x, y: p.y });
              gsap.set(".h-ring", { opacity: p.ring });
              gsap.set(".h-corona", { opacity: p.corona });
              gsap.set(".h-diamond", { opacity: p.diamond });
            },
          });
        });
      } else {
        const htl = gsap.timeline({
          defaults: { ease: "none", duration: 1 },
          scrollTrigger: { trigger: ".steps", start: "top 55%", end: "bottom 55%", scrub: 0.6 },
        });
        for (let i = 1; i < PH.length; i++) {
          const p = PH[i], at = i - 1;
          htl.to(".h-moon", { x: p.x, y: p.y }, at)
             .to(".h-ring", { opacity: p.ring }, at)
             .to(".h-corona", { opacity: p.corona }, at)
             .to(".h-diamond", { opacity: p.diamond, scale: p.diamond ? 1 : 0.4, transformOrigin: "50% 50%" }, at);
        }
        steps.forEach((s, i) => {
          ScrollTrigger.create({
            trigger: s, start: "top 60%", end: "bottom 60%",
            onToggle: (self) => self.isActive && setPhase(i),
          });
        });

        /* Reveals (skipped in ?settle screenshot mode) */
        document.querySelectorAll(".section .kicker, .section .h2, .section .lede, .step, .video").forEach((el) => el.classList.add("reveal"));
        if (!SETTLE) gsap.set(".reveal", { autoAlpha: 0, y: 36 });
        ScrollTrigger.batch(".reveal", {
          start: "top 88%",
          once: true,
          onEnter: (els) => gsap.to(els, { autoAlpha: 1, y: 0, duration: 0.9, stagger: 0.08, ease: "power3.out", overwrite: true }),
        });

        /* Result counters and bars */
        if (!SETTLE) gsap.utils.toArray("[data-count]").forEach((el) => {
                    gsap.from(el, {
            textContent: el.dataset.from ? +el.dataset.from : 0,
            duration: 1.6, ease: "power2.out", snap: { textContent: 1 },
            scrollTrigger: { trigger: el, start: "top 85%", once: true },
          });
        });
        if (!SETTLE) gsap.from(".bar__fill", {
          scaleX: 0, duration: 1.2, stagger: 0.15, ease: "power2.out",
          scrollTrigger: { trigger: ".bars", start: "top 85%", once: true },
        });
      }
      setPhase(0);

      /* Desktop-only: magnetic CTAs and a soft cursor glow */
      if (!reduce && matchMedia("(pointer: fine)").matches) {
        const cleanups = [];
        document.querySelectorAll(".hero__ctas .btn--primary, #dl-btn").forEach((btn) => {
          btn.setAttribute("data-magnetic", "");
          const xTo = gsap.quickTo(btn, "x", { duration: 0.5, ease: "power3.out" });
          const yTo = gsap.quickTo(btn, "y", { duration: 0.5, ease: "power3.out" });
          const move = (e) => {
            const r = btn.getBoundingClientRect();
            const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
            const reach = Math.max(r.width, r.height) * 0.9;
            if (Math.abs(dx) < r.width / 2 + reach && Math.abs(dy) < r.height / 2 + reach) { xTo(dx * 0.28); yTo(dy * 0.36); }
            else { xTo(0); yTo(0); }
          };
          window.addEventListener("pointermove", move, { passive: true });
          cleanups.push(() => { window.removeEventListener("pointermove", move); gsap.set(btn, { x: 0, y: 0 }); });
        });
        const glow = document.querySelector(".cursor-glow");
        if (glow) {
          const gx = gsap.quickTo(glow, "x", { duration: 0.7, ease: "power3.out" });
          const gy = gsap.quickTo(glow, "y", { duration: 0.7, ease: "power3.out" });
          const mv = (e) => { document.documentElement.classList.add("has-cursor"); gx(e.clientX); gy(e.clientY); };
          window.addEventListener("pointermove", mv, { passive: true });
          cleanups.push(() => { window.removeEventListener("pointermove", mv); document.documentElement.classList.remove("has-cursor"); });
        }
        return () => cleanups.forEach((f) => f());
      }
    }
  );

  // Images and fonts change layout; recompute trigger positions once loaded.
  window.addEventListener("load", () => ScrollTrigger.refresh());
})();
