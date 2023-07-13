import tensorflow as tf
import numpy as np
print(tf.test.gpu_device_name())

a = tf.constant([1, 3, 3, 5])
b = tf.constant([2, 2, 2, 2])

print(f"test {tf.multiply(a, b)}")

print(tf.random.experimental.stateless_split((np.random.random_integers(0, 512), np.random.random_integers(0, 512)), num=1)[0, :])

print(tf.random.experimental.stateless_split((np.random.random_integers(0, 512), np.random.random_integers(0, 521)), num=1)[0, :])

print(tf.random.experimental.stateless_split((np.random.random_integers(0, 512), np.random.random_integers(0, 512)), num=1)[0, :])