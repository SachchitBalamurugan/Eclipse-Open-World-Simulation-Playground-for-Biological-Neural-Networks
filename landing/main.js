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
      btn.href = "downloads/" + encodeURIComponent(file).replace(/%2F/g, "/");
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
      if (reduce) {
        // Static totality with a visible diamond-ring point; no movement.
        gsap.set(".e-moon, .e-rim", { x: 0, y: 0 });
        gsap.set(".e-halo", { opacity: 0.12 });
        gsap.set(".e-diamond", { scale: 0.45, opacity: 0.9 });
        gsap.set(".e-corona", { opacity: 1, scale: 1 });
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
        if (/[?&]settle/.test(location.search)) { tl.delay(0).pause(); tl.progress(1); }
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
        if (!/[?&]settle/.test(location.search)) gsap.set(".reveal", { autoAlpha: 0, y: 28 });
        ScrollTrigger.batch(".reveal", {
          start: "top 88%",
          once: true,
          onEnter: (els) => gsap.to(els, { autoAlpha: 1, y: 0, duration: 0.8, stagger: 0.08, overwrite: true }),
        });

        /* Result counters and bars */
        if (!/[?&]settle/.test(location.search)) gsap.utils.toArray("[data-count]").forEach((el) => {
                    gsap.from(el, {
            textContent: el.dataset.from ? +el.dataset.from : 0,
            duration: 1.6, ease: "power2.out", snap: { textContent: 1 },
            scrollTrigger: { trigger: el, start: "top 85%", once: true },
          });
        });
        if (!/[?&]settle/.test(location.search)) gsap.from(".bar__fill", {
          scaleX: 0, duration: 1.2, stagger: 0.15, ease: "power2.out",
          scrollTrigger: { trigger: ".bars", start: "top 85%", once: true },
        });
      }
      setPhase(0);
    }
  );

  // Images and fonts change layout; recompute trigger positions once loaded.
  window.addEventListener("load", () => ScrollTrigger.refresh());
})();
