#Computation
#d = 81
python3 compute_solution.py --config_experiment 'Experiment_2' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0 --Rx 80.0 --precomputed_input 'no' --lambda_smoothing 0.1 --eta_coef_photon 10000000 --enforce_smooth_u

#d = 85
python3 compute_solution.py --config_experiment 'Experiment_3' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0 --Rx 80.0 --precomputed_input 'no' --lambda_smoothing 0.1 --eta_coef_photon 10000000 --enforce_smooth_u

#d = 60
python3 compute_solution.py --config_experiment 'Experiment_4' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0 --Rx 80.0 --precomputed_input 'no' --lambda_smoothing 0.1 --eta_coef_photon 10000000 --enforce_smooth_u

#d = 100
python3 compute_solution.py --config_experiment 'Experiment_5' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0 --Rx 80.0 --precomputed_input 'no' --lambda_smoothing 0.1 --eta_coef_photon 10000000 --enforce_smooth_u

#Evaluation
#d = 81
python3 evaluate_solution.py --config_experiment 'Experiment_2' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0

#d = 85
python3 evaluate_solution.py --config_experiment 'Experiment_3' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0

#d = 60
python3 evaluate_solution.py --config_experiment 'Experiment_4' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0

# d = 100
python3 evaluate_solution.py --config_experiment 'Experiment_5' --N_photon 44.0 --compute_mult 'no' --compute_proton 'no' --compute_photon 'yes'  --N1 44.0 --N2 0.0