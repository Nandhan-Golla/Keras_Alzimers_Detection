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
  
  def __init__(self, n_components= 2, kernel= 'rbf', gamma=None):
    self.n_components = n_components
    self.kernel = kernel
    self.gamma = gamma
    self.scaler = StandardScaler()
    self.kcpa = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma)


  def _to_numpy_(self, X):
    if isinstance(X, tf.data.Dataset):
      
      images = []
      for batch_images, batch_labels in X:
        images.append(batch_images.numpy())
      X = np.concatenate(images, axis=0)
    elif isinstance(X, tf.Tensor):
      X = X.numpy()
    elif isinstance(X, list):
      X = np.array(X)
    return X
  
  def _flatten_images_(self, X):
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)
  
  def fit(self, X, y =None):
    X = self._to_numpy_(X)
    X = X/255.0
    X_flat = self._flatten_images_(X)
    X_scaled = self.scaler.fit_transform(X_flat)
    x_pca = self.kcpa.fit(X_scaled)

    return self
  
  def transform(self, X):
    X = self._to_numpy_(X)
    X = X/255.0
    X_flat = self._flatten_images_(X)
    X_scaled = self.scaler.transform(X_flat)
    x_kcpa = self.kcpa.transform(X_scaled)
    return x_kcpa
  
try:
  Dta = IterativeTransformer()
  Dta.fit(train_data)
  Dta.transform(train_data)
except Exception as e :
     print(f"Nahh {e}")
  


  





