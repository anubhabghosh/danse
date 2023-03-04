#!/bin/bash
for r in 10.0
do
	python3.7 main_danse_gs.py \
	--mode train \
	--rnn_model_type lstm \
	--dataset_type LorenzSSM \
	--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_1000_N_500_r2_$(echo $r)dB_nu_-20.0dB.pkl \
	--splits ./data/synthetic_data/splits_m_3_n_3_LorenzSSM_data_T_1000_N_500_r2_$(echo $r)dB_nu_-20.0dB.pkl
done
