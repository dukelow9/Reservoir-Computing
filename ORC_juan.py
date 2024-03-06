import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras import layers, initializers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

class EchoStateNetwork(layers.Layer):
    def __init__(self, neurons, decay=0.1, alpha=0.5, spectralRad=0.9, scale=1.0, seed=None, epsilon=None, sparseness=0.0,
                 activation=None, optimize=False, optimizeVars=None, *args, **kwargs):
        self.seed = seed
        self.neurons = neurons
        self.state_size = neurons
        self.sparseness = sparseness
        self.decay = decay
        self.alpha = alpha
        self.spectralRad = spectralRad
        self.scale = scale
        self.epsilon = epsilon
        self._activation = tf.tanh if activation is None else activation
        self.optimize = optimize
        self.optimizeVars = optimizeVars

        super(EchoStateNetwork, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.optimize_table = {var: var in self.optimizeVars for var in ["alpha", "spectralRad", "decay", "scale"]}
        self.decay = self.add_weight(initializer=initializers.constant(self.decay),
                                     trainable=self.optimize_table["decay"], dtype=tf.float32)
        self.alpha = self.add_weight(initializer=initializers.constant(self.alpha),
                                     trainable=self.optimize_table["alpha"], dtype=tf.float32)
        self.spectralRad = self.add_weight(initializer=initializers.constant(self.spectralRad),
                                   trainable=self.optimize_table["spectralRad"], dtype=tf.float32)
        self.scale = self.add_weight(initializer=initializers.constant(self.scale),
                                  trainable=self.optimize_table["scale"], dtype=tf.float32)
        self.storeAlpha = self.add_weight(initializer=initializers.constant(self.alpha),
                                           trainable=False, dtype=tf.float32)
        self.ratio = self.add_weight(initializer=initializers.constant(1.0),
                                          trainable=False, dtype=tf.float32)
        self.kernel = self.add_weight(shape=(input_shape[-1], self.neurons), 
                                      initializer=initializers.RandomUniform(-1, 1, seed=self.seed), trainable=False)
        self.recurrent_kernel_init = self.add_weight(shape=(self.neurons, self.neurons),
                                                     initializer=initializers.RandomNormal(seed=self.seed), trainable=False)
        self.recurrent_kernel = self.add_weight(shape=(self.neurons, self.neurons), initializer=tf.zeros_initializer(),
                                                trainable=False)
        self.recurrent_kernel_init.assign(self.setSparseness(self.recurrent_kernel_init))
        self.recurrent_kernel.assign(self.setAlpha(self.recurrent_kernel_init))
        self.ratio.assign(self.echoStateRatio(self.recurrent_kernel))
        self.spectralRad.assign(self.findEchoStatespectralRad(self.recurrent_kernel * self.ratio))

        self.built = True

    def setAlpha(self, W):
        return 0.5 * (self.alpha * (W + tf.transpose(W)) + (1 - self.alpha) * (W - tf.transpose(W)))

    def setSparseness(self, W):
        mask = tf.cast(tf.random.uniform(W.shape, seed=self.seed) < (1 - self.sparseness), dtype=W.dtype)
        return W * mask

    def echoStateRatio(self, W):
        eigvals = tf.linalg.eigvals(W)
        return tf.reduce_max(tf.abs(eigvals))

    def findEchoStatespectralRad(self, W):
        target = 1.0
        eigvals = tf.linalg.eigvals(W)
        x = tf.math.real(eigvals)
        y = tf.math.imag(eigvals)
        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        sol = (tf.sqrt(b**2 - 4*a*c) - b)/(2*a)

        spectralRad = tf.reduce_min(sol)
        return spectralRad

    def call(self, inputs, states):
        rkernel = self.setAlpha(self.recurrent_kernel_init)
        if self.alpha != self.storeAlpha:
            self.decay.assign(tf.clip_by_value(self.decay, 0.00000001, 0.25))
            self.alpha.assign(tf.clip_by_value(self.alpha, 0.000001, 0.9999999))
            self.spectralRad.assign(tf.clip_by_value(self.spectralRad, 0.5, 1.0e100))
            self.scale.assign(tf.clip_by_value(self.scale, 0.5, 1.0e100))

            self.ratio.assign(self.echoStateRatio(rkernel))
            self.spectralRad.assign(self.findEchoStatespectralRad(rkernel * self.ratio))
            self.storeAlpha.assign(self.alpha)

        ratio = self.spectralRad * self.ratio * (1 - self.epsilon)
        previousOutput = states[0]
        output = previousOutput + self.decay * (
                tf.matmul(inputs, self.kernel * self.scale) +
                tf.matmul(self._activation(previousOutput), rkernel * ratio)
                - previousOutput)

        return self._activation(output), [output]
    
def convertToBinaryPattern(data):
    threshold = np.mean(data)
    binaryPattern = np.where(data > threshold, 1, 0)
    return binaryPattern

def opticalDiffuser(inputPattern, transmissionMatrix):
    diffusedPattern = np.dot(inputPattern, transmissionMatrix)
    return diffusedPattern
    
earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
    
#################################################################################################################

def generateMackeyGlass(length, beta=0.2, gamma=0.1, n=10, tau=25, noise_strength=0.02):
    x = 1.2 * np.ones((length + tau,))
    for t in range(tau, length + tau - 1):
        x[t + 1] = x[t] + (beta * x[t - tau]) / (1 + x[t - tau]**n) - gamma * x[t]
        x[t + 1] += noise_strength * np.random.normal()
    return x[tau:]

length = 2025
mackeyGlass = generateMackeyGlass(length)

steps = 50
activation = tf.keras.activations.relu

x = np.array([mackeyGlass[i:i+steps] for i in range(length - steps)])
y = np.array([mackeyGlass[i+steps] for i in range(length - steps)])

print(x.shape)
x = x.reshape((x.shape[0], -1))
print(x.shape)
print(y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, shuffle=False)
xTrain, xValid, yTrain, yValid = train_test_split(xTrain, yTrain, test_size=0.1, shuffle=False)

cell = EchoStateNetwork(neurons=100, activation=activation, decay=0.3, epsilon=1e-20, alpha=0.1, optimize=True,
                        optimizeVars=["spectralRad", "decay", "alpha", "scale"], seed=np.random.randint(0, 1000))
recurrentLayer = tf.keras.layers.RNN(cell, input_shape=(steps, 1), return_sequences=False, name="nn")
output = tf.keras.layers.Dense(1, name="readouts")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model = tf.keras.models.Sequential()
model.add(recurrentLayer)
model.add(output)
model.compile(loss="mse", optimizer=optimizer)
model.summary()

binaryMackeyGlass = convertToBinaryPattern(mackeyGlass)
transmissionMatrix = np.random.rand(steps, steps)
transmissionMatrix = np.eye(steps,steps)
# transmissionMatrix = np.random.randn(steps, steps)/2 + 1j*np.random.randn(steps, steps)/2

print(x.shape)
print(xTrain.shape)
print(yTrain.shape)
print(transmissionMatrix.shape)

xTrainDiffused = opticalDiffuser(xTrain, transmissionMatrix)
xValidDiffused = opticalDiffuser(xValid, transmissionMatrix)
xTestDiffused = opticalDiffuser(xTest, transmissionMatrix)

# plt.figure(1); plt.clf
# plt.subplot(2,1,1)
# plt.plot(mackeyGlass, label="Mackey Glass")

# plt.subplot(2,1,2)
# plt.plot(binaryMackeyGlass, label="Binary Mackey Glass")
# # plt.show()

# plt.figure(2); plt.clf
# plt.subplot(2,2,1)
# plt.pcolor(xTrain)

# plt.subplot(2,2,2)
# plt.pcolor(xTrainDiffused)
# plt.show()

startTime = time.time()

history = model.fit(xTrainDiffused, yTrain, epochs=2000,
                    validation_data=(xValidDiffused, yValid), callbacks=[earlyStopping])

endTime = time.time()

predictions = model.predict(xTestDiffused.reshape((xTestDiffused.shape[0], -1)))

elapsedTime = endTime - startTime
print(f"Training Time: {elapsedTime} seconds")

##################################################################################################################

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

loss = model.evaluate(xTestDiffused.reshape((xTestDiffused.shape[0], -1)), yTest)
print(f"Test Loss: {loss}")

binaryMackeyGlass = binaryMackeyGlass[:45 * 45].reshape((45, 45))

plt.figure(figsize=(8, 8))
plt.imshow(binaryMackeyGlass.T, cmap='summer', aspect='auto', extent=[0, 45, 0, 45])
plt.title('Binary Mackey Glass (Heatmap)')
plt.xlabel('Time (t)')
plt.ylabel('Pattern Index')
plt.show()
