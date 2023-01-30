#!/bin/bash
python3.8 generate_data.py \
--n_states 10 \
--n_obs 10 \
--num_samples 200 \
--sequence_length 6000 \
--inverse_r2_dB 0 \
--dataset_type LinearSSM \
--output_path ./data/synthetic_data/ \