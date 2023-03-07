#!/bin/bash
python3.8 generate_data.py \
--n_states 3 \
--n_obs 3 \
--num_samples 500 \
--sequence_length 1000 \
--inverse_r2_dB 20 \
--nu_dB -20 \
--dataset_type LorenzSSM \
--output_path ./data/synthetic_data/ \