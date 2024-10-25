#!/bin/zsh

mkdir -p out
results_dir="out"

# Define your custom list of distances
distance_list=(0 5 10 15 20 25 30 35 40 45 50)

for d in "${distance_list[@]}"; do
  config_file="configs/distance/d${d}.json"
  result_path="${results_dir}/d${d}"

  echo "$(date +%T) Running simulation for distance d = ${d} m"
  (python mapc_dcf/run.py -c "$config_file" -r "$result_path" | cat) > "${results_dir}/d${d}.log" 2>&1
done