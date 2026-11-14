#!/usr/bin/env bash
# Fan the NeuroRover benchmark out over Freestyle VMs (runs from the lead laptop only).
#
#   npx -y freestyle@latest login          # once, opens the browser
#   bash sweep/run_sweep.sh launch 20 30   # 20 VMs, 30 seeds  (extra args go to bench.py)
#   bash sweep/run_sweep.sh status
#   bash sweep/run_sweep.sh collect        # pull results/*.json into neurorover/results/
#   bash sweep/run_sweep.sh teardown       # delete every nr-sweep-* VM
#
# Cost: default VM = 4 vCPU / 8 GiB ~ $0.27/h. Free tier caps running VMs at 10 (LIMIT_EXCEEDED past that); stack shards per VM.
set -euo pipefail
cd "$(dirname "$0")/.."
FS="npx -y freestyle@latest"
STATE=sweep/vms.txt
TARBALL=sweep/payload.tar.gz
DATA=resources/challenges/freeze-preparation/09-freeze/data.json.gz

vm_id_from_json() {  # read create --output json on stdin, print the VM id
  python -c 'import json,sys
d=json.load(sys.stdin)
for src in (d, d.get("vm") or {}):
    v=src.get("id") or src.get("vmId")
    if v: print(v); break'
}

launch() {
  local nvm=${1:-20} seeds=${2:-30}; shift 2 || true
  local extra="$*"
  [ -f "$DATA" ] || { echo "missing $DATA (clone the hackathon resources repo into resources/)"; exit 1; }
  git archive --format=tar HEAD -o sweep/payload.tar
  tar -rf sweep/payload.tar "$DATA"
  gzip -f sweep/payload.tar
  : > "$STATE"
  for i in $(seq 0 $((nvm - 1))); do
    (
      id=$($FS vm create --slug "nr-sweep-$i" --replace-slug --no-ssh --idle-timeout-seconds 7200 \
            --metadata project=neurorover --output json | vm_id_from_json)
      echo "$i $id" >> "$STATE"
      $FS vm scp "$TARBALL" "$id:/home/ubuntu/payload.tar.gz" >/dev/null
      $FS vm exec "$id" --timeout-ms 280000 -- bash -lc "
        set -e; mkdir -p /home/ubuntu/nr && cd /home/ubuntu/nr && tar -xzf /home/ubuntu/payload.tar.gz
        python3 -m venv .venv && .venv/bin/pip -q install numpy
        nohup .venv/bin/python -m neurorover.bench --all --seeds $seeds --procs \$(nproc) \
              --shard $i --nshards $nvm $extra > bench.log 2>&1 &
        echo started shard $i on \$(nproc) cpus"
    ) &
  done
  wait
  sort -n "$STATE" -o "$STATE"
  echo "launched $(wc -l < "$STATE") VMs -> $STATE"
}

status() {
  while read -r i id; do
    n=$($FS vm exec "$id" -- bash -lc "ls /home/ubuntu/nr/neurorover/results 2>/dev/null | wc -l; pgrep -f neurorover.bench >/dev/null && echo running || echo done" | tr '\n' ' ')
    echo "shard $i ($id): $n"
  done < "$STATE"
}

collect() {
  mkdir -p neurorover/results
  while read -r i id; do
    $FS vm exec "$id" -- bash -lc "cd /home/ubuntu/nr/neurorover && tar -czf /home/ubuntu/results.tgz results" >/dev/null
    $FS vm scp "$id:/home/ubuntu/results.tgz" "sweep/results_$i.tgz" >/dev/null
    tar -xzf "sweep/results_$i.tgz" -C neurorover/
  done < "$STATE"
  echo "results now: $(ls neurorover/results | wc -l) files"
}

teardown() {
  while read -r i id; do $FS vm delete "$id" >/dev/null && echo "deleted $id"; done < "$STATE"
}

"${1:-launch}" "${@:2}"
