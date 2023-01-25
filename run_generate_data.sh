#!/bin/bash
python3.8 generate_data.py \
--n_states 5 \
--n_obs 5 \
--num_samples 100 \
--sequence_length 50 \
--dataset_type LinearSSM \
--output_path ./data/synthetic_data/