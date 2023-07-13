import tensorflow as tf
import numpy as np
import math

# limit ram
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

# Optimizations
# GPU
tf.config.optimizer.set_jit(True)
# CPU?
tf.function(jit_compile=True)
# Mixed precision
# not compatible with trained weights of the efficient net
tf.keras.mixed_precision.set_global_policy('mixed_float16')

(X, Y), (X_Test, Y_Test) = tf.keras.datasets.cifar10.load_data()

# Explore data
print(X.shape)
print(Y.shape)
print(Y[0])

# Split dataset into train and validation
m = X.shape[0]
n_train = math.ceil(m * 0.8)
np.random.RandomState(42).shuffle(X)
np.random.RandomState(42).shuffle(Y)
X_Train = X[:n_train]
Y_Train = Y[:n_train]
X_Val = X[n_train:]
Y_Val = Y[n_train:]

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    brightness_range=(0.25, 1.25),
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    vertical_flip=True)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),

    tf.keras.layers.Conv2D(64, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, 3, padding="valid"),
    tf.keras.layers.Activation("relu"),

    tf.keras.layers.GlobalMaxPool2D(),

    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),

    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"],
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

print(model.summary())


def scheduler(epoch, lr):
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# Define callbacks
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint", save_only_best=True,
                                                     save_weights_only=True, verbose=1)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.fit(datagen_train.flow(X_Train, Y_Train, batch_size=512),
     validation_data=datagen_val.flow(X_Val, Y_Val, batch_size=128),
     epochs=500, callbacks=[save_best_model, early_stop_callback])

print(f"Evaluation metrics last model: {model.evaluate(datagen_test.flow(X_Test, Y_Test))}")

model.load_weights("checkpoint")
print(f"Evaluation metrics best model's weights: {model.evaluate(datagen_test.flow(X_Test, Y_Test))}")

model.save(filepath="cfar10_model.h5", save_format="h5")
