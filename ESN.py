import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class EchoStateNetwork(keras.layers.Layer):
    def __init__(self, reservoirSize, spectralRadius=0.9, sparsity=0.2, seed=None, **kwargs):
        super(EchoStateNetwork, self).__init__(**kwargs)
        self.reservoirSize = reservoirSize
        self.spectralRadius = spectralRadius
        self.sparsity = sparsity
        self.seed = seed

    def build(self, inputShape):
        self.reservoirWeights = self.add_weight(
            "reservoirWeights",
            shape=(inputShape[-1], self.reservoirSize),
            initializer=tf.initializers.random_normal(mean=0, stddev=0.1, seed=self.seed),
            trainable=False,
        )
        self.recurrentWeights = self.add_weight(
            "recurrentWeights",
            shape=(self.reservoirSize, self.reservoirSize),
            initializer=tf.initializers.orthogonal(gain=self.spectralRadius, seed=self.seed),
            trainable=False,
        )
        self.bias = self.add_weight(
            "bias",
            shape=(self.reservoirSize,),
            initializer="zeros",
            trainable=False,
        )
        super(EchoStateNetwork, self).build(inputShape)

    def call(self, inputs):
        x = tf.matmul(inputs, self.reservoirWeights) + self.bias
        x = tf.tanh(tf.matmul(x, self.recurrentWeights))
        return x

def generateData(batch_size, steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(batch_size, steps) - 0.5)
    return series[..., np.newaxis].astype(np.float32)

np.random.seed(42)
steps = 50
series = generateData(500, steps + 1)
xTrain, xTest, yTrain, yTest = train_test_split(series[:, :steps], series[:, -1], test_size=0.2, shuffle=False)
xTrain, xValid, yTrain, yValid = train_test_split(xTrain, yTrain, test_size=0.1, shuffle=False)

esn = EchoStateNetwork(reservoirSize=100, spectralRadius=0.9, sparsity=0.2, seed=42)

model = keras.models.Sequential([esn, keras.layers.Dense(1, activation='linear', use_bias=False)])
optimizer = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss="mse", optimizer="adam")

xTrain = xTrain.reshape((xTrain.shape[0], -1))
xValid = xValid.reshape((xValid.shape[0], -1))

history = model.fit(xTrain, yTrain, epochs=100, validation_data=(xValid, yValid))

predictions = model.predict(xTest.reshape((xTest.shape[0], -1)))
loss = model.evaluate(xTest.reshape((xTest.shape[0], -1)), yTest)
print(f"Test Loss: {loss}")
