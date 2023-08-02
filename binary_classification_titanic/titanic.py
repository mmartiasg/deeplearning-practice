import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

BATCH_SIZE = 128
with tf.device("CPU") as d:
    X, Y = tfds.as_numpy(tfds.load(
        'titanic',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))

features = [str(f) for f in X.keys()]

features_array = []
for f in features:
    features_array.append(X[f])

X_dataset = np.column_stack(features_array)
