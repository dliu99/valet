#!/bin/zsh
cd "$(dirname "$0")/.."
for side in right left; do
  out=$(timeout 900 npm run valet -- do "Pick up the glass pill jar with a silver lid (it may be lying on its side) and set it down about 10 cm to the $side." --what "glass pill jar with a silver lid" --steps 30 2>&1)
  echo "$out" | grep -E "^run #|^  -> " | cut -c1-300
done
echo BATCH DONE
