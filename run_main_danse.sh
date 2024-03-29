#!/bin/bash
python3.8 main_danse.py \
--mode train \
--rnn_model_type gru \
--dataset_type LorenzSSM \
--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_r2_40.0dB_nu_-20dB.pkl \
--splits ./data/synthetic_data/splits.pkl