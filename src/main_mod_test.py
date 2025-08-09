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
  
  def __init__(self, n_components= 2, kernel= 'rbf', gamma=None, max_samples=1000):
    self.n_components = n_components
    self.kernel = kernel
    self.gamma = gamma
    self.max_samples = max_samples
    self.scaler = StandardScaler()
    self.kcpa = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma)


  def _to_numpy_sample_(self, X, max_samples=None):

    if max_samples is None:
      max_samples = self.max_samples
      
    if isinstance(X, tf.data.Dataset):
      images = []
      samples_collected = 0
      print(f"Extracting up to {max_samples} samples from dataset...")
      
      for batch_images, batch_labels in X:
        batch_array = batch_images.numpy()
        remaining_samples = max_samples - samples_collected
        if remaining_samples <= 0:
          break
        
        if batch_array.shape[0] > remaining_samples:
          batch_array = batch_array[:remaining_samples]
        
        images.append(batch_array)
        samples_collected += batch_array.shape[0]
        
        if samples_collected >= max_samples:
          break
      
      if images:
        X = np.concatenate(images, axis=0)
        print(f"Successfully extracted {X.shape[0]} samples")
      else:
        raise ValueError("No images found in dataset")
    elif isinstance(X, tf.Tensor):
      X = X.numpy()
      if len(X) > max_samples:
        X = X[:max_samples]
    elif isinstance(X, list):
      X = np.array(X)
      if len(X) > max_samples:
        X = X[:max_samples]
    elif isinstance(X, np.ndarray):
      if len(X) > max_samples:
        X = X[:max_samples]
    
    return X
  
  def _flatten_images_(self, X):
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)
  
  def fit(self, X, y =None):
    print(f"Fitting IterativeTransformer with max {self.max_samples} samples...")
    X = self._to_numpy_sample_(X)
    print(f"Loaded {X.shape[0]} samples with shape {X.shape}")
    X = X/255.0
    X_flat = self._flatten_images_(X)
    print(f"Flattened shape: {X_flat.shape}")
    print("Fitting StandardScaler...")
    X_scaled = self.scaler.fit_transform(X_flat)
    print("Fitting KernelPCA...")
    x_pca = self.kcpa.fit(X_scaled)
    print("Fitting complete!")

    return self
  
  def transform(self, X):
    print("Transforming data...")
    X = self._to_numpy_sample_(X)
    print(f"Loaded {X.shape[0]} samples for transformation")
    X = X/255.0
    X_flat = self._flatten_images_(X)
    X_scaled = self.scaler.transform(X_flat)
    x_kcpa = self.kcpa.transform(X_scaled)
    print(f"Transformation complete! Output shape: {x_kcpa.shape}")
    return x_kcpa
  
try:

  Dta = IterativeTransformer(max_samples=1000)
  Dta.fit(train_data)
  result = Dta.transform(train_data)
  print(f"Final result shape: {result.shape}")
  print("Success! The error has been fixed.")
except Exception as e:
     print(f"Error occurred: {e}")
