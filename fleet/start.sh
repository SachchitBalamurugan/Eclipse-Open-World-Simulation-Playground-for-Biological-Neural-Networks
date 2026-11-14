#!/bin/bash
# start.sh <shard> <nshards> <condition> <seeds> [tag]
set -e
SH=${1#s}; NS=${2#n}; SEEDS=${4#k}
pkill -f neurorover.bench 2>/dev/null || true
mkdir -p /home/ubuntu/nr && cd /home/ubuntu/nr && tar -xzf /home/ubuntu/payload.tar.gz
[ -x .venv/bin/python ] || python3 -m venv .venv; .venv/bin/pip -q install numpy
TAG=""; [ -n "$5" ] && TAG="--tag $5"
nohup .venv/bin/python -m neurorover.bench --condition "$3" --seeds "$SEEDS" --procs 4 --shard "$SH" --nshards "$NS" $TAG > bench.log 2>&1 &
echo "started shard $SH/$NS on $(nproc) cpus"
