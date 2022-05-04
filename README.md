# OFDM-simulator-with-ML-equalizer
All the code and data used for my Bachelor's thesis "OFDM channel equalization with machine learning" are available here.

ML OFDM estimator directory cointains the ML models, as well as the main.py file used to train the network.
OFDM_sim_withML.m contains the matlab code with the ml estimator included.
OFDM_sim_NoML.m is the same OFDM simulator without the ml estimator.

A word of warning: The simulator with the ml estimator is extremely slow with large data amounts, due to the inefficiency of the intermittent estimate.py script.
If you simply want to try the simulator, using the version without the ml estimator is recommended.
