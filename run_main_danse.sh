#!/bin/bash
python3.8 main_danse.py \
--mode train \
--model_type gru \
--dataset_type LorenzSSM \
--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_6000_N_200_r2_40.0dB.pkl \
--splits None 