#!/bin/zsh

mkdir -p out

for i in {1..10}; do
  config_file="configs/collisions_vs_density/n${i}.json"
  result_path="out/n${i}"

  (python mapc_dcf/run.py -c "$config_file" -r "$result_path" | cat) > "out/n${i}.log" 2>&1 &
done