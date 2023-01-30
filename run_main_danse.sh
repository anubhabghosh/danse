#!/bin/bash
python3.8 main_danse.py \
--mode train \
--rnn_model_type gru \
--dataset_type LinearSSM \
--datafile ./data/synthetic_data/trajectories_m_5_n_5_LinearSSM_data_T_6000_N_200_r2_40.0dB.pkl \
--splits ./data/synthetic_data/splits.pkl