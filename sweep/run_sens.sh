#!/usr/bin/env bash
# Learning-rate sensitivity for the plastic cultures on Freestyle.
# One VM per (condition, lr); 10 seeds each, 4 procs per VM. Results are tagged
# (e.g. culture_rstdp_lr1_s3.json) so analysis/summarize.py keeps them out of the headline table.
#   bash sweep/run_sens.sh launch | collect | teardown
set -euo pipefail
cd "$(dirname "$0")/.."
FS="npx -y freestyle@latest"
STATE=sweep/vms_sens.txt
TARBALL=sweep/payload.tar.gz   # built by run_sweep.sh launch
LRS="0.2 1 5"
CONDS="culture_rstdp culture_fep"
launch() {
  : > "$STATE"
  for c in $CONDS; do for lr in $LRS; do
    (
      tag="lr${lr}"; slug="nr-sens-${c#culture_}-${tag//./p}"
      id=$($FS vm create --slug "$slug" --replace-slug --no-ssh --idle-timeout-seconds 7200 --output json | python -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("id") or d.get("vmId"))')
      echo "$slug $id" >> "$STATE"
      $FS vm scp "$TARBALL" "$id:/home/ubuntu/payload.tar.gz" >/dev/null
      $FS vm exec "$id" --timeout-ms 280000 -- bash -lc "
        set -e; mkdir -p ~/nr && cd ~/nr && tar -xzf ~/payload.tar.gz
        python3 -m venv .venv && .venv/bin/pip -q install numpy
        nohup .venv/bin/python -m neurorover.bench --condition $c --seeds 10 --procs \$(nproc) \
          --culture-kw '{\"lr\": $lr}' --tag $tag > bench.log 2>&1 &
        echo started $c $tag"
    ) &
  done; done
  wait; echo "launched $(wc -l < "$STATE") VMs"
}
collect() {
  mkdir -p neurorover/results
  while read -r slug id; do
    $FS vm exec "$id" -- bash -lc "cd ~/nr/neurorover && tar -czf ~/results.tgz results" >/dev/null
    $FS vm scp "$id:/home/ubuntu/results.tgz" "sweep/results_$slug.tgz" >/dev/null
    tar -xzf "sweep/results_$slug.tgz" -C neurorover/
  done < "$STATE"
  ls neurorover/results | grep -c "_lr" | xargs echo "tagged result files:"
}
teardown() { while read -r slug id; do $FS vm delete "$id" >/dev/null && echo "deleted $slug"; done < "$STATE"; }
"${1:-launch}"
