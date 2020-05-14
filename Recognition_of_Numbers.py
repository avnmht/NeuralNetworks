import tensorflow
import random
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
i=0
data=keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=data.load_data()
train_images=train_images/255.0
test_images=test_images/255
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
    ])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,train_labels,epochs=5)
prediction=model.predict(test_images)

while i<10:
    p=random.randint(0,300)
    plt.xlabel(np.argmax(prediction[p]))
    plt.imshow(test_images[p],cmap=plt.cm.binary)
    plt.show()
    i+=1
