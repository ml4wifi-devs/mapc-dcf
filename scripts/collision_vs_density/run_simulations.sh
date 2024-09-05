#!/bin/zsh

for i in {1..10}; do
  config_file="mapc_dcf/configs/density_increase/n${i}.json"
  result_path="out/n${i}"

  python mapc_dcf/run.py -c "$config_file" -r "$result_path" > "simulation_${i}.log" &
done