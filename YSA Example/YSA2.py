import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  

model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'),   
    Dense(64, activation='relu'),   
    Dense(10, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test doğruluğu:', test_acc)


#Olmayan kütüphaneleri import etmeyi unutmayınız. Saygılarımala Z_#
