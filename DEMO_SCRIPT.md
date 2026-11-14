# NeuroRover demo script (2 minutes)

## Hook (20 seconds)

Companies like Cortical Labs rent living neurons and say they learn from far fewer examples than silicon. Testing that costs about $300 a week. Most of the work isn't the neurons, it's the plumbing. So we built the whole closed-loop pipeline with no living cells, behind the real CL API, so the claim can be tested cheaply.

## Demo (40 seconds)

Open http://localhost:8765.

1. Point at the left panel, "64-electrode culture." That's a simulated 8x8 electrode array with 1,024 neurons.
2. Read the colours. Blue is sensory input: the weather features become pulses. Green votes DEPLOY, amber votes HOLD. Pink is feedback and purple is reward. Grey is the rest of the culture.
3. Pick `culture_reservoir` in the dropdown and press Play.
4. Point at the middle panel. The robot decides whether to cover crops tonight, from weather 3 to 7 days back.
5. Point at the scoreboard on the right. It checks each night against whether it really froze.

## The figure (30 seconds)

Show `neurorover/analysis/fig_learning_curves.png`. Validation AUC at 25 to 200 training samples, 10 seeds.

- The plastic cultures stay near chance, the same wall two public DishBrain replications hit.
- The reservoir culture ties logistic regression at 25 samples but falls behind at 100 and up.

## Honest limitation (15 seconds)

A simulation can't tell you if real neurons are more sample-efficient. What we have is a harness that can.

## Next step (15 seconds)

A week on Cortical Cloud with `run_cl.py --hardware`. Same encoding, same 300 validation nights, same budgets. If it fails, we learn that for $300.

## Likely questions

**Why freeze data, not soil?** We tried soil moisture first. Calibration error barely correlated with any feature (|r| < 0.16), so every condition would have sat at chance.

**Why trust the simulator?** Its activity looks like a culture: about 5 Hz per neuron and a burst every 2.5 s. Stimulation raises firing 3 to 7x over sham. It isn't fit to real recordings, so we don't claim it predicts real neurons.

**Isn't this just logistic regression?** For the reservoir, mostly yes. The culture acts as a random feature map and the silicon readout does the learning. That's a finding, not something we hid.

**What's the CL API path?** The culture is a `cl.sim.SimulatorDataSource`, so the loop uses the same calls as a CL1. Through the API, reservoir AUC was 0.89 against 0.87 in the fast harness on the same 100 rows.

**How many seeds?** Ten per condition, with bootstrap confidence intervals. The cloud sweep for more seeds is written but didn't run because of billing setup.

**What would change your mind?** A living culture with DishBrain-style feedback beating the frozen control and getting near logistic at 25 samples. That would be the first real evidence for the claim.
