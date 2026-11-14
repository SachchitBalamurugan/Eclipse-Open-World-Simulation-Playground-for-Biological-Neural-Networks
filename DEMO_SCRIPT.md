# NeuroRover demo script (about 3 minutes)

Before judges arrive:
- Open the desktop app: `app\start.bat`.
- Serve the web pages: `.venv/Scripts/python -m http.server 8765 -d demo`.
- Check no VMs are running: `npx -y freestyle@latest vm list`. Only launch a fleet if you want the live wall; 10 VMs cost about $2.70/h. Tear down afterwards.

## Hook (20 seconds)

Companies like Cortical Labs rent out living neurons and say they learn from far fewer examples than silicon. Testing that costs about $300 a week per dish, and most of the work isn't the neurons, it's the plumbing. So we built the whole closed loop with no living cells. A simulated culture sits behind the real Cortical Labs API, it drives a frost-protection robot on the hackathon's NOAA freeze data, and we benchmark it against normal models.

## The app (60 seconds)

1. **Simulate tab.** The dish in the middle is the culture: 1,024 model neurons on a 64-electrode array.
   - Blue electrodes are sensory: tonight's weather features become stimulation pulses.
   - Amber votes DEPLOY, teal votes HOLD.
   - Green beams are the teaching stimulation.
2. **The field.** Click Robot. The rover is built from real parts (Raspberry Pi 5, SHT45 sensor, Pololu motors, Actuonix actuator); hover any part to see its spec. On a freeze night it drives out and the cover deploys. The fence, crates and drums are NVIDIA SimReady assets.
3. **Right panel.** Rolling accuracy and weight drift: the sensory-to-motor weights change as it's taught. The learning curve at the bottom compares it with every baseline.
4. **Fleet tab** (only if VMs are up). Each tile is a cloud VM training a different seed. Pop one out into its own window.

## The result (40 seconds)

Show `neurorover/analysis/fig_learning_curves.png` or the app's learning-curve panel. AUC on 300 held-out nights, 50 seeds.

- **Reward-only learning doesn't work.** DishBrain-style feedback and reward-modulated STDP stay near chance, the same wall two public DishBrain replications hit.
- **Paired-stimulation teaching works.** Replay the input, then stimulate the correct motor region a few milliseconds later. The culture goes from 0.55 to 0.73 AUC, +0.24 over the untrained culture at 200 nights. It's stimulation only, so a real dish could be taught the same way.
- **Imagined nights help.** Training first on 150 nights imagined by our weather world model adds about +0.11 at 25 real nights.
- **Silicon still wins.** Logistic regression gets 0.89 at 200. The culture learns, but it's slower.

## Honest limitation (15 seconds)

A simulation can't tell you whether real neurons are more sample-efficient. What we built is the harness that can, with the experiment already written.

## Next step (15 seconds)

A week on Cortical Cloud with `run_cl.py --hardware`: same encoding, same teaching protocol, same 300 nights, same sample budgets. If living neurons beat this curve, that's real evidence. If they don't, we learn it for $300.

## Likely questions

**Why freeze data, not soil?** We tried the soil data first. Calibration error barely correlated with any feature (|r| < 0.16), so every method sat at chance.

**Why trust the simulator?** Its activity looks like a culture: about 5 Hz per neuron, a network burst every 2.5 s, and stimulation raises local firing about 9x over sham. It isn't fit to real recordings, so we don't claim it predicts real neurons.

**Isn't teaching just supervised learning?** Yes: the label decides which motor region gets stimulated. The point is that the label goes in as stimulation, which a real dish can receive. Reward-only signals didn't work in our model.

**What's the CL API path?** The culture is a `cl.sim.SimulatorDataSource`, so the loop uses the same calls as a CL1. Through the API, reservoir AUC matched the fast harness (0.89 vs 0.87 on the same rows).

**How many seeds?** 50 per condition (10 local, 40 on Freestyle cloud VMs), with paired bootstrap confidence intervals.

**What would change your mind?** A living culture taught this way that beats our simulated curve at 25 to 50 samples. That would be the first real evidence for the sample-efficiency claim.
