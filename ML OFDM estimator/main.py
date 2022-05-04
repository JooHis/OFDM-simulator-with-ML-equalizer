import sys
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf
from tensorflow.python.keras.layers import *
from sklearn import model_selection
from keras.layers import BatchNormalization
to_categorical = tf.keras.utils.to_categorical


def create_model():
    # Feedforward Neural Network Model
    model = tf.keras.Sequential()

    # Layers
    model.add(tf.keras.layers.Flatten(input_shape=(4,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='softmax'))


    # Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model


def ofdm_ml_estimator():


    # Load in the train and test data
    train_data = scipy.io.loadmat(r'training_data_8pilots_chEst_highSNR.mat')
    test_data = scipy.io.loadmat(r'test_data_8pilots_chEst_allSNR.mat')
    train_data = train_data["train_data_8pilots"]
    test_data = test_data["test_data_8pilots"]

    train_received = train_data[0:len(train_data), 0:4]
    train_transmitted = train_data[0:len(train_data), 4]
    test_received = test_data[0:len(test_data), 0:4]
    test_transmitted = test_data[0:len(test_data), 4]


    # Create model
    model = create_model()


    # Split train data and validation data. Fit model
    train_received, val_received, train_transmitted, val_transmitted = model_selection.train_test_split(train_received, train_transmitted, test_size=0.2)

    history = model.fit(train_received, train_transmitted, validation_data=(val_received, val_transmitted), epochs=20, batch_size=64, verbose=1)

    # Save model
    #model.save('ML_estimator_16pilots')


    # Test accuracy
    _, acc = model.evaluate(test_received, test_transmitted, verbose=1)
    print(f'The accuracy of predictions is {round(acc * 100, 3)}')


    # Show plot and save to file
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'categorical_crossentropy'], loc='upper left')
    plt.show()
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


if __name__ == '__main__':
    ofdm_ml_estimator()



