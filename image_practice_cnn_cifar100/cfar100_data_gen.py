import tensorflow as tf
import numpy as np
import math

# limit ram
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])

# Optimizations
tf.keras.mixed_precision.set_global_policy('mixed_float16')

(X, Y), (X_Test, Y_Test) = tf.keras.datasets.cifar100.load_data()

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
    brightness_range=(0.25, 1.55),
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

    # no drop after convs
    #0.8 74
    #0.6 0.75
    #0.4 77
    #0.25 77
    # no dropout
    # 1024 fcl
    # Evaluation metrics best model's weights: [2.669071912765503, 0.32020002603530884, 0.7571000456809998]
    # 4096 fcl
    # [2.67095947265625, 0.32360002398490906, 0.7605000138282776]
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(4096, activation="relu"),
    tf.keras.layers.Dropout(rate=0.4),

    tf.keras.layers.Dense(100, activation="softmax")
])

top_10_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k=10, name='sparse_top_k_categorical_accuracy', dtype=None
)

model.compile(loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy", top_10_accuracy],
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              jit_compile=True)

print(model.summary())


# Define callbacks
def scheduler(epoch, lr):
    if epoch < 15:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
# val_sparse_top_k_categorical_accuracy or val_loss
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath="model_temporal_checkpoint", save_only_best=True,
                                                     save_weights_only=True, verbose=1)

model.fit(
    datagen_train.flow(X_Train, Y_Train, batch_size=512, shuffle=True),
    validation_data=datagen_val.flow(X_Val, Y_Val, batch_size=128),
    epochs=500,
    callbacks=[save_best_model, early_stop_callback, reduce_lr]
)

print(f"Evaluation metrics last model: {model.evaluate(datagen_test.flow(X_Test, Y_Test))}")

model.load_weights("model_temporal_checkpoint")
print(f"Evaluation metrics best model's weights: {model.evaluate(datagen_test.flow(X_Test, Y_Test))}")

model.save(filepath="cfar100_model.h5", save_format="h5")
