# Eclipse — brand spec (shared by the app and the landing page)

**Product name:** Eclipse. **Descriptor:** "Biological neural network robot studio".
**Slogan (exact casing):** "A New ERA of training Biological Neural Networks" — use as the hero headline on the landing
page, in the app's About/splash and under the header wordmark where space allows. "ERA" stays all caps.
**One-liner:** Train any robot with biological neural network controllers — in simulation, across a fleet of VMs, then export the protocol to a living culture.
**Honesty line (always visible somewhere):** "Controllers are simulated Izhikevich neuron cultures — no living neurons are used in simulation."

## Concept
An eclipse is a living, biological light (the sun's corona) revealed only when something dark passes in front of it.
Eclipse the product is mostly darkness (totality) with one warm, glowing ring of light: the corona = the neural culture.
Use: deep near-black surfaces, a single warm corona accent, soft radial glows (sparingly), thin rings and arcs.
The 6-step workflow maps to eclipse phases, drawn as small disc icons where a dark moon progressively covers a gold sun:
1 Import (first contact) · 2 Configure (partial) · 3 Platform (partial, deeper) · 4 Train (totality — corona ring) ·
5 Evaluate (diamond ring) · 6 Deploy (third contact / sun returning).

## Palette
| token | hex | use |
|---|---|---|
| --void | #050507 | app/page background (totality) |
| --surface-1 | #0B0B10 | panels |
| --surface-2 | #12121A | raised rows, inputs |
| --line | #22222C | 1px borders |
| --line-strong | #33333F | focused borders |
| --text | #EDEAF2 | primary text |
| --text-dim | #8C8A99 | secondary text |
| --corona | #F2B45A | primary accent (buttons, active step, links) |
| --corona-hi | #FFD48A | hover / highlight |
| --corona-glow | rgba(242,180,90,0.18) | glows, focus rings |
| --chromosphere | #FF6A4D | errors / stop / danger (the red solar limb) |
| --plasma | #8FA6FF | info, secondary data series, "simulated" badges |
| --ok | #6FD3A8 | success states (tested-mock, done) |
| --warn | #E8C25A | warnings (sdk-untested) |

Data viz series order: corona, plasma, ok, chromosphere. Spike heatmap: void → #3A2A12 → corona → #FFF3D6.

## Logo
A dark disc (moon) offset slightly up-right over a gold ring (corona), with a tiny bright point at ~1 o'clock
(the "diamond ring"). SVG, works at 16px. Wordmark: "ECLIPSE" in Inter 600, letter-spacing 0.32em, uppercase.

## Type & shape
Inter (UI, 400/500/600), JetBrains Mono (numbers, logs). Corners 4px (panels), 999px (pills/badges).
Section headers: 11px uppercase, letter-spacing 0.14em, --text-dim. Buttons: primary = --corona fill, #1A1206 text.

## Motion (GSAP 3.13)
Slow, celestial: ease "power2.out"/"sine.inOut", 0.4–0.9s for UI, longer for hero. The corona may breathe
(subtle scale/opacity loop) during training/totality only. Always respect prefers-reduced-motion via gsap.matchMedia.

## Not allowed
No NVIDIA logos/names as branding. No claims of living neurons in simulation. No fake download counts or testimonials.
