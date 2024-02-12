import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1  

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='sgd', loss='mse')

history = model.fit(X, y, epochs=100, verbose=0)

plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()


#Olmayan kütüphaneleri import etmeyi unutmayınız. Saygılarımala Z_#
