import os
import numpy as np
import tensorflow as tf


def estimate_channel(OFDM_symbol, pilots):
    predictions = []
    DIR = os.getcwd()
    if (int(pilots) == 8):
        est = tf.keras.models.load_model(DIR + '/ML OFDM estimator/ML_estimator_8pilots')
    elif (int(pilots) == 16):
        est = tf.keras.models.load_model(DIR + '/ML OFDM estimator/ML_estimator_16pilots')
    else:
        print("Number of pilots must be 8 or 16!")

    OFDM_symbol = np.array(OFDM_symbol).reshape(64 - int(pilots), 1, 4)
    for i in range(0, len(OFDM_symbol)):
        y = est.predict(OFDM_symbol[i])
        y = int(np.argmax(y))
        predictions.append(y)
    return predictions


z = estimate_channel(OFDM_symbol, pilots)