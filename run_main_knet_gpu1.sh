#!/bin/bash
for r in -25.0 -20.0 -10.0 0.0 10.0 20.0
do
	python3.7 main_kalmannet.py \
	--mode train \
	--knet_model_type KNetUoffline \
	--dataset_type LorenzSSM \
	--datafile ./data/synthetic_data/trajectories_m_3_n_3_LorenzSSM_data_T_100_N_500_r2_$(echo $r)dB_nu_-20.0dB.pkl \
	--splits ./data/synthetic_data/splits.pkl
done
