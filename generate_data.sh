#!/bin/bash

mkdir data/new_sim/periodic
name=new_sim/periodic

for i in {1..100}; do
    python3 generate_data.py --seed=$i --name=$name --mech=periodic --noise_scale=3
done