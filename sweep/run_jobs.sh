#!/usr/bin/env bash
# Run arbitrary training jobs, one per Freestyle VM, all at once.
#   bash sweep/run_jobs.sh launch sweep/jobs.txt   # jobs.txt lines: <slug>|<shell command run from repo root>
#   bash sweep/run_jobs.sh status
#   bash sweep/run_jobs.sh collect                 # pulls sim/runs and surgical/results back
#   bash sweep/run_jobs.sh teardown
# Free tier: at most 10 running VMs (~$0.27/h each). Always collect before teardown.
set -euo pipefail
cd "$(dirname "$0")/.."
FS="npx -y freestyle@latest"
STATE=sweep/jobs_state.txt
TARBALL=sweep/jobs_payload.tar.gz
DATA=resources/challenges/freeze-preparation/09-freeze/data.json.gz

build_payload() {
  # working tree (tracked + untracked code), minus heavy/irrelevant dirs, plus the freeze data
  { git ls-files; git ls-files --others --exclude-standard; } | sort -u \
    | grep -v -E '^(app/|demo/|runs/|fleet/|assets/simready/hf/|neurorover/results/|neurorover/tuning/results/)' \
    | grep -v -E '\.(tgz|log|pkl|zip|mp4|gif)$' > sweep/jobs_files.txt
  echo "$DATA" >> sweep/jobs_files.txt
  tar -czf "$TARBALL" -T sweep/jobs_files.txt
  echo "payload $(du -h "$TARBALL" | cut -f1), $(wc -l < sweep/jobs_files.txt) files"
}

launch() {
  local jobs=${1:?jobs file}
  build_payload
  : > "$STATE"
  while IFS='|' read -r slug cmd; do
    [ -z "$slug" ] && continue
    (
      id=$($FS vm create --slug "$slug" --replace-slug --no-ssh --idle-timeout-seconds 10800 \
            --metadata project=neurorover --output json | python -c 'import json,sys
d=json.load(sys.stdin); print(d.get("id") or d.get("vmId"))')
      echo "$slug $id" >> "$STATE"
      for t in 1 2 3; do $FS vm scp "$TARBALL" "$id:/home/ubuntu/payload.tar.gz" >/dev/null && break || sleep 5; done
      printf '%s\n' "set -e" "mkdir -p ~/nr && cd ~/nr && tar -xzf ~/payload.tar.gz" "$cmd" "touch ~/nr/JOB_DONE" > /tmp/job_$slug.sh
      $FS vm scp "/tmp/job_$slug.sh" "$id:/home/ubuntu/job.sh" >/dev/null
      $FS vm exec "$id" --timeout-ms 60000 -- bash -lc "nohup bash ~/job.sh > ~/job.log 2>&1 & echo started $slug"
    ) &
  done < "$jobs"
  wait
  sort "$STATE" -o "$STATE"
  echo "launched $(wc -l < "$STATE") VMs -> $STATE"
}

status() {
  while read -r slug id; do
    s=$($FS vm exec "$id" -- bash -lc "test -f ~/nr/JOB_DONE && echo DONE || echo running; tail -n 2 ~/job.log" 2>&1 | tr '\n' ' ' | cut -c1-220)
    echo "$slug: $s"
  done < "$STATE"
}

collect() {
  mkdir -p sweep/collected
  while read -r slug id; do
    $FS vm exec "$id" -- bash -lc "cd ~/nr && tar -czf ~/out.tgz --ignore-failed-read sim/runs surgical/results ~/job.log 2>/dev/null; true" >/dev/null
    $FS vm scp "$id:/home/ubuntu/out.tgz" "sweep/collected/$slug.tgz" >/dev/null && echo "collected $slug" || echo "FAILED $slug"
  done < "$STATE"
}

teardown() { while read -r slug id; do $FS vm delete "$id" >/dev/null && echo "deleted $slug"; done < "$STATE"; }

"${1:-status}" "${@:2}"
