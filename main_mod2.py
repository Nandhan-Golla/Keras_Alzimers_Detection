import keras as ke 
import tensorflow as tf 
import os 
import dotenv as env 


env.load_dotenv()
data_tarin = ke.utils.image_dataset_from_directory(os.environ('DATA_TRAIN_SET_PATH'))
data_val = ke.utils.image_dataset_from_directory(os.environ('DATA_VAL_SET_PATH'))

model_save_path = os.environ('MODEL_PATH')
data_tarin = data_tarin.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
data_val = data_val.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))

model = ke.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights=None, classes=2)
model.compile(optimizer=ke.optimizers.Adam(learning_rate=0.0001),
              loss=ke.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(data_tarin, validation_data=data_val, epochs=10)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
print("Training completed successfully.")