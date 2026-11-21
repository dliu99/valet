#!/bin/zsh
# Alternating pick-move-place goals with Claude checkpoints; one summary line per run.
cd "$(dirname "$0")/.."
for side in "$@"; do
  out=$(timeout 900 npm run valet -- do "Pick up the white supplement bottle, move it about 10 cm to the $side, and set it down standing upright." --what "white supplement bottle" --steps 30 2>&1)
  echo "$out" | grep -E "^run #|^  -> " | cut -c1-300
done
echo BATCH DONE
