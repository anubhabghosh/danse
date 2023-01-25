#!/bin/bash
python3.8 main_danse.py \
--mode train \
--model_type gru \
--dataset_type LinearSSM \
--datafile ./data/synthetic_data/trajectories_m_5_n_5_LinearSSM_data_T_50_N_100_r2_40.0dB.pkl \
--splits None 