import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras import layers, initializers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

class OpticalReservoir(layers.Layer):
    def __init__(self, transmissionMatrix, activation=None, *args, **kwargs):
        self.transmissionMatrix = transmissionMatrix
        self._activation = tf.tanh if activation is None else activation
        super(OpticalReservoir, self).__init__(*args, **kwargs)

    def call(self, inputs):
        output = tf.matmul(inputs, self.transmissionMatrix)
        return self._activation(output)

class modelTraining:
    def __init__(self, activation, transmissionMatrix):
        self.activation = activation
        self.transmissionMatrix = transmissionMatrix
        self.earlyStopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    def generateMackeyGlass(self, length, beta=0.2, gamma=0.1, n=10, tau=25, noiseStrength=0.02):
        x = 1.2 * np.ones((length + tau,))
        for t in range(tau, length + tau - 1):
            x[t + 1] = x[t] + (beta * x[t - tau]) / (1 + x[t - tau]**n) - gamma * x[t]
            x[t + 1] += noiseStrength * np.random.normal()
        return x[tau:]

    def trainAndVisualize(self):
        length = 4000
        mackeyGlass = self.generateMackeyGlass(length)
        steps = 50

        x = np.array([mackeyGlass[i:i+steps] for i in range(length - steps)])
        y = np.array([mackeyGlass[i+steps] for i in range(length - steps)])

        x = x.reshape((x.shape[0], -1))
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, shuffle=False)
        xTrain, xValid, yTrain, yValid = train_test_split(xTrain, yTrain, test_size=0.1, shuffle=False)

        opticalReservoir = OpticalReservoir(transmissionMatrix=self.transmissionMatrix, activation=self.activation)
        opticalReservoir.trainable = False
        output = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), name="readouts")

        optimizer = tf.keras.optimizers.Adam()

        model = tf.keras.models.Sequential()
        model.add(opticalReservoir)
        model.add(output)
        model.build(input_shape=(None, steps))
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()

        startTime = time.time()
        history = model.fit(xTrain, yTrain, epochs=2000,
                            validation_data=(xValid, yValid), callbacks=[self.earlyStopping])
        endTime = time.time()

        predictions = model.predict(xTest.reshape((xTest.shape[0], -1)))

        elapsedTime = endTime - startTime
        print(f"Training Time: {elapsedTime} seconds")
        
        plt.figure(figsize=(10, 5))
        plt.title("Model Predictions")
        plt.plot(yTest, label="Actual Data", linestyle='--', color='navy', linewidth='1.5')
        plt.plot(predictions, label="Prediction Data", color='yellowgreen')
        plt.xlabel("Time (t)")
        plt.ylabel("Intensity (Wm$^-2$)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.plot(history.history['loss'], label='Training Loss', color='mediumpurple')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        loss = model.evaluate(xTest.reshape((xTest.shape[0], -1)), yTest)
        print(f"Test Loss: {loss}")

activation = tf.keras.activations.relu
dim = 50
transmissionMatrix = np.random.randn(dim, dim)/2 + 1j * np.random.randn(dim, dim)/2

training = modelTraining(activation, transmissionMatrix)
training.trainAndVisualize()
