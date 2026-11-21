#!/bin/zsh
# Pick-move-place goals given as "object:direction" pairs; one summary line per step.
cd "$(dirname "$0")/.."
while pgrep -f "brain/cli.ts do" >/dev/null; do sleep 5; done
for pair in "$@"; do
  what=${pair%%:*}; side=${pair##*:}
  out=$(timeout 900 npm run valet -- do "Pick up the $what, move it about 10 cm to the $side, and set it down standing upright." --what "$what" --steps 30 2>&1)
  echo "$out" | grep -E "^run #|^  -> " | cut -c1-300
done
echo BATCH DONE
