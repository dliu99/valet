#!/bin/zsh
# Alternating pick-move-place goals for the white box; one summary line per step.
cd "$(dirname "$0")/.."
for side in "$@"; do
  out=$(timeout 900 npm run valet -- do "Pick up the white box, move it about 10 cm to the $side, and set it down standing upright." --what "white box" --steps 30 2>&1)
  echo "$out" | grep -E "^run #|^  -> " | cut -c1-300
done
echo BATCH DONE
