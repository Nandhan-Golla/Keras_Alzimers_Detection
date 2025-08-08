import keras
import tensorflow as tf 
from matplotlib import pyplot as plt
import dotenv as env
import os

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.base import BaseEstimator
env.load_dotenv()

data_tr = keras.utils.image_dataset_from_directory(os.environ.get("DATA_TRAIN_SET_PATH"))
data_vl = keras.utils.image_dataset_from_directory(os.environ.get("DATA_VAL_SET_PATH"))

data_tr = data_tr.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
data_vl = data_vl.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.save(os.environ.get("MODEL_PATH"))
history = model.fit(data_tr, epochs=10, validation_data=data_vl)
print(history.history)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
