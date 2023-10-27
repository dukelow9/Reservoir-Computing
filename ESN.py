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
            initializer = tf.initializers.glorot_uniform(seed=self.seed),
            trainable = False,
        )
        self.recurrentWeights = self.add_weight(
            "recurrentWeights",
            shape = (self.reservoirSize, self.reservoirSize),
            initializer = tf.initializers.orthogonal(gain=self.spectralRadius, seed=self.seed),
            trainable = False,
        )
        self.bias = self.add_weight(
            "bias",
            shape = (self.reservoirSize,),
            initializer = "zeros",
            trainable = False,
        )
        super(EchoStateNetwork, self).build(inputShape)

    def call(self, inputs):
        x = tf.tanh(tf.matmul(inputs, self.reservoirWeights) + self.bias)
        x = tf.tanh(tf.matmul(x, self.recurrentWeights))
        return x

dataLength = 2000
t = np.linspace(0, 60 * np.pi, dataLength)
opticalData = (0.5 * np.sin(0.1 * t) + 0.2 * np.sin(0.5 * t) + 0.3 * np.sin(1.0 * t) + 0.1 * np.random.randn(dataLength))

sequenceLength = 500
X, Y = [], []

for i in range(dataLength - sequenceLength):
    X.append(opticalData[i:i + sequenceLength])
    Y.append(opticalData[i + sequenceLength])

X = np.array(X)
Y = np.array(Y)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, shuffle=False)

reservoirSize = 400
esn = EchoStateNetwork(reservoirSize = reservoirSize)

outputLayer = keras.layers.Dense(1, activation = "linear")

inputData = keras.layers.Input(shape=(sequenceLength,))
ESNOutput = esn(inputData)
output = outputLayer(ESNOutput)

model = keras.models.Model(inputs=inputData, outputs=output)
model.compile(optimizer = "adam", loss = "mean_squared_error")

epochCount = 10
for epoch in range(epochCount):
    history = model.fit(xTrain, yTrain, epochs=1, verbose=0)
    trainLoss = history.history['loss'][0]
    testLoss = model.evaluate(xTest, yTest, verbose=0)
    print(f"Epoch {epoch + 1}/{epochCount}  ---  Train Loss: {trainLoss:.4f}  ---  Test Loss: {testLoss:.4f}")

predictions = model.predict(xTest)
testLoss = model.evaluate(xTest, yTest)
print(f"Test Loss: {testLoss:.4f}")
