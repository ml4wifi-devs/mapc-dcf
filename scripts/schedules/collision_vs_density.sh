#!/bin/zsh

mkdir -p out
results_dir="out"

for i in {1..10}; do
  config_file="configs/collisions_vs_density/n${i}.json"
  result_path="${results_dir}/n${i}"

  echo "$(date +%T) Running simulation for N=${i}"
  (python mapc_dcf/run.py -c "$config_file" -r "$result_path" | cat) > "${results_dir}/n${i}.log" 2>&1
done