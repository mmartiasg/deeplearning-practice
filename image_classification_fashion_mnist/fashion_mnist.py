import tensorflow as tf
import math

(X, Y), (X_Test, Y_Test) = tf.keras.datasets.fashion_mnist.load_data()

X = X.reshape(-1, 28, 28, 1)
X_Test = X_Test.reshape(-1, 28, 28, 1)

CLASSES = 10
WIDTH=HEIGHT=28

# DataGen
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    brightness_range=(0.25, 1.25),
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    vertical_flip=True)

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

# Split
N = X.shape[0]
n_train = math.ceil(N * 0.8)

X_Train = X[:n_train]
Y_Train = Y[:n_train]

X_Val = X[n_train:]
Y_Val = Y[n_train:]

# model
model = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=HEIGHT, width=WIDTH),
    #GRAYSCALE
    tf.keras.Input(shape=(WIDTH, HEIGHT, 1)),

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

    tf.keras.layers.Dense(CLASSES, activation="softmax")
])

# compile model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy],
              jit_compile=True)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temporal_checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

model.fit(datagen_train.flow(X_Train, Y_Train, batch_size=128),
validation_data=datagen_val.flow(X_Val, Y_Val, batch_size=64),
epochs=500, callbacks=[model_checkpoint, early_stop])

model.load_weights("temporal_checkpoint")

model.save(filepath="fashion_mnist.h5", save_format="h5")

print(f"Eval on test: {model.evaluate(datagen_test.flow(X_Test, Y_Test, batch_size=64))}")