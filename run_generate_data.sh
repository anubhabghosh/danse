#!/bin/bash
python3.8 generate_data.py \
--n_states 2 \
--n_obs 2 \
--num_samples 2 \
--sequence_length 100 \
--inverse_r2_dB 20 \
--nu_dB 0 \
--dataset_type SinusoidalSSM \
--output_path ./data/synthetic_data/ \