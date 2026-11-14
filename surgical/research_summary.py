"""Aggregate independently seeded research VM evaluations without selecting a winner."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def main():
    files = sorted((ROOT/'results').glob('research_vm_*.json'))
    if not files:
        raise SystemExit('No collected research VM JSONs')
    runs = [json.loads(p.read_text()) for p in files]
    seeds = [r['seed'] for r in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate run seeds; refusing to double count')
    rows = [row for run in runs for row in run['rows']]
    report = ['# Procedural surgical scenario evaluation', '',
              f'{len(runs)} VM runs; seeds {seeds}. Simulated Izhikevich culture; no living neurons or hardware.', '',
              'These are reach-control perturbation experiments inspired by the cited papers, not reproductions of their surgical tasks.', '',
              '| Scenario | Controller | Episodes | Success | Median error (mm) | Episodes with contacts |',
              '|---|---|---:|---:|---:|---:|']
    summary=[]
    for scenario, controller in sorted({(r['scenario'],r['controller']) for r in rows}):
        group=[r for r in rows if r['scenario']==scenario and r['controller']==controller]
        stats=dict(scenario=scenario,controller=controller,n=len(group),
                   success_rate=float(np.mean([r['success'] for r in group])),
                   median_error_mm=float(np.median([r['final_err_mm'] for r in group])),
                   contact_episode_rate=float(np.mean([r['collisions']>0 for r in group])))
        summary.append(stats)
        report.append(f"| {scenario} | {controller} | {len(group)} | {stats['success_rate']:.1%} | {stats['median_error_mm']:.1f} | {stats['contact_episode_rate']:.1%} |")
    report += ['', 'Each controller sees matched scenario seeds and targets. Contact counts use the existing convex collision geometry; they are not tissue-injury measures.',
               'Small episode counts are exploratory. Fixed culture checkpoints are evaluated without selecting a learning rate on these targets.', '',
               'See individual JSON files for dynamics-model metrics, disturbances, and per-episode outcomes.']
    (ROOT/'results/research_summary.json').write_text(json.dumps(dict(seeds=seeds,files=[p.name for p in files],summary=summary),indent=2))
    (ROOT/'RESEARCH_RESULTS.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))

if __name__ == '__main__': main()
