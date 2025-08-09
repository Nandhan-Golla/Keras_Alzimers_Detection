import os 
from sklearn.decomposition import KernelPCA
#from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import dotenv as env
import keras as ke
import tensorflow as tf 
from matplotlib import pyplot 
import numpy as np 
env.load_dotenv()

gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

train_data = ke.utils.image_dataset_from_directory(os.environ.get("DATA_TRAIN_SET_PATH"))
validation_data = ke.utils.image_dataset_from_directory(os.environ.get("DATA_VAL_SET_PATH"))


class IterativeTransformer(BaseEstimator, TransformerMixin):
  
  def __init__(self, n_components= 2, kernel= 'rbf', gamma=None, batch_size=1000):
    self.n_components = n_components
    self.kernel = kernel
    self.gamma = gamma
    self.batch_size = batch_size
    self.scaler = StandardScaler()
    self.kcpa = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma)
    self.is_fitted = False


  def _get_dataset_batches_(self, X):
    """Generator that yields batches of data from the dataset"""
    if isinstance(X, tf.data.Dataset):
      current_batch = []
      current_size = 0
      total_processed = 0
      
      for batch_images, batch_labels in X:
        batch_array = batch_images.numpy()
        batch_labels_array = batch_labels.numpy()
        
        for i in range(batch_array.shape[0]):
          current_batch.append((batch_array[i], batch_labels_array[i]))
          current_size += 1
          total_processed += 1
          
          if current_size >= self.batch_size:
            images = np.array([item[0] for item in current_batch])
            labels = np.array([item[1] for item in current_batch])
            yield images, labels, total_processed
            current_batch = []
            current_size = 0
      
      if current_batch:
        images = np.array([item[0] for item in current_batch])
        labels = np.array([item[1] for item in current_batch])
        yield images, labels, total_processed
    
    elif isinstance(X, (tf.Tensor, np.ndarray, list)):
      if isinstance(X, tf.Tensor):
        X = X.numpy()
      elif isinstance(X, list):
        X = np.array(X)
      
      total_samples = len(X)
      for i in range(0, total_samples, self.batch_size):
        end_idx = min(i + self.batch_size, total_samples)
        yield X[i:end_idx], None, end_idx
    
    else:
      raise ValueError(f"Unsupported data type: {type(X)}")
  
  def _flatten_images_(self, X):
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)
  
  def fit(self, X, y=None):
    print(f"Fitting IterativeTransformer with batch size {self.batch_size}...")
    
    first_batch = True
    all_scaled_data = []
    batch_count = 0
    total_samples = 0
    
    for batch_images, batch_labels, processed_so_far in self._get_dataset_batches_(X):
      batch_count += 1
      batch_size = batch_images.shape[0]
      total_samples += batch_size
      
      print(f"Processing batch {batch_count}, samples: {batch_size}, total processed: {processed_so_far}")
      
      batch_images = batch_images / 255.0
      batch_flat = self._flatten_images_(batch_images)
      
      if first_batch:
        print(f"First batch shape: {batch_images.shape}, flattened: {batch_flat.shape}")
        print("Fitting StandardScaler on first batch...")
        batch_scaled = self.scaler.fit_transform(batch_flat)
        first_batch = False
      else:
        batch_scaled = self.scaler.transform(batch_flat)
      
      all_scaled_data.append(batch_scaled)
    
    print(f"Processed {batch_count} batches with {total_samples} total samples")
    

    print("Combining all batches for KernelPCA fitting...")
    X_all_scaled = np.vstack(all_scaled_data)
    print(f"Combined data shape: {X_all_scaled.shape}")
    
    print("Fitting KernelPCA on all data...")
    self.kcpa.fit(X_all_scaled)
    self.is_fitted = True
    print("Fitting complete!")
    
    return self
  
  def transform(self, X):
    if not self.is_fitted:
      raise ValueError("Transformer must be fitted before transform")
    
    print(f"Transforming data in batches of {self.batch_size}...")
    
    all_transformed_data = []
    batch_count = 0
    total_samples = 0
    
    for batch_images, batch_labels, processed_so_far in self._get_dataset_batches_(X):
      batch_count += 1
      batch_size = batch_images.shape[0]
      total_samples += batch_size
      
      print(f"Transforming batch {batch_count}, samples: {batch_size}, total processed: {processed_so_far}")

      batch_images = batch_images / 255.0
      batch_flat = self._flatten_images_(batch_images)
      

      batch_scaled = self.scaler.transform(batch_flat)
      batch_transformed = self.kcpa.transform(batch_scaled)
      
      all_transformed_data.append(batch_transformed)
    
    print(f"Transformed {batch_count} batches with {total_samples} total samples")
    
    result = np.vstack(all_transformed_data)
    print(f"Transformation complete! Final output shape: {result.shape}")
    return result
  
try:

  Dta = IterativeTransformer(batch_size=1000)
  Dta.fit(train_data)
  result = Dta.transform(train_data)
  print(f"Final result shape: {result.shape}")
  print("Success! The transformer now processes the full dataset in batches!")
except Exception as e:
     print(f"Error occurred: {e}")



model = ke.models.Sequential([
    ke.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    ke.layers.MaxPooling2D((2, 2)),
    ke.layers.Dense(64, activation='relu', kernel_regularizer=ke.regularizers.l2(0.01)),
    ke.layers.Conv2D(64, (3, 3), activation='relu'),
    ke.layers.MaxPooling2D((2, 2)),
    ke.layers.Conv2D(128, (3, 3), activation='relu'),
    ke.layers.MaxPooling2D((2, 2)),
    ke.layers.Flatten(),
    ke.layers.Dense(128, activation='relu'),
    ke.layers.Dense(10, activation='relu'),
    ke.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
hist = model.fit(train_data, epochs=15, validation_data=validation_data)
model.save(os.environ.get("MODEL_PATH"))