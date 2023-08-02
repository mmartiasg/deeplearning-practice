import tensorflow_datasets as tfds
import tensorflow as tf
import math
import re
import numpy as np

# limit ram to run multiple algorithms at once
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4196)])

with tf.device("CPU") as d:
    X, Y = tfds.as_numpy(tfds.load(
        'sentiment140',
        split='train',
        batch_size=-1,
        as_supervised=True,
    ))
    X_test, Y_test = tfds.as_numpy(tfds.load(
        'sentiment140',
        split='test',
        batch_size=-1,
        as_supervised=True,
    ))

# X = np.array([re.sub(r'[0-9]+', '[NUMBER]', str(x)) for x in X])
# X_test = np.array([re.sub(r'[0-9]+', '[NUMBER]', str(x)) for x in X_test])

# vectorizer and padding
OOV_TOKEN = "[UNK]"
PADDING = "post"
TRUNCATING = "post"
VOCABULARY_SIZE = 30000
MAX_SIZE = 200
# 0 = negative, 2 = neutral, 4 = positive
CLASSES = 3
# from 0 to 2 after sustitution

# I need to sustitute 2 by 1 and 4 by 3
# to be able to use an sparse categorical crossentropy
# or else I will need to use 5 classes and never use 1 and 3
Y[Y == 2] = 1
Y[Y == 4] = 2
Y_test[Y_test == 2] = 1
Y_test[Y_test == 4] = 2

vectorizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE, oov_token=OOV_TOKEN, split=" ",
                                                   char_level=False)
padding = tf.keras.preprocessing.sequence.pad_sequences

# Split
N = X.shape[0]
n_train = math.ceil(N * 0.9)

X_train = X[:n_train]
Y_train = Y[:n_train]

X_val = X[n_train:]
Y_val = Y[n_train:]

# fit on train!
vectorizer.fit_on_texts([str(t) for t in X_train])

X_train = padding(vectorizer.texts_to_sequences([str(t) for t in X_train]), padding=PADDING, truncating=TRUNCATING,
                  maxlen=MAX_SIZE)
X_val = padding(vectorizer.texts_to_sequences([str(t) for t in X_val]), padding=PADDING, truncating=TRUNCATING,
                maxlen=MAX_SIZE)
X_test = padding(vectorizer.texts_to_sequences([str(t) for t in X_test]), padding=PADDING, truncating=TRUNCATING,
                 maxlen=MAX_SIZE)

# Build datasets
train_datasets = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(4096).batch(4096).prefetch(
    tf.data.AUTOTUNE)
val_datasets = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(1024).prefetch(tf.data.AUTOTUNE)
test_datasets = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(512).prefetch(tf.data.AUTOTUNE)

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCABULARY_SIZE, output_dim=32, mask_zero=True),
    tf.keras.layers.GRU(units=32, recurrent_dropout=0.2,
                        return_sequences=False),
    tf.keras.layers.Dense(units=CLASSES, activation="softmax")
])

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy],
              jit_compile=True)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temporal_checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

# fit model
model.fit(train_datasets, validation_data=val_datasets, epochs=15, callbacks=[early_stop, model_checkpoint])

model.load_weights("temporal_checkpoint")

print(f"model evaluated on test: {model.evaluate(test_datasets)}")

# 5625/5625 [==============================] - 123s 22ms/step - loss: 0.3823 - sparse_categorical_accuracy: 0.8331 - val_loss: 0.5477 - val_sparse_categorical_accuracy: 0.7638
# 4/4 [==============================] - 0s 114ms/step - loss: 4.8562 - sparse_categorical_accuracy: 0.5361
# [4.856170177459717, 0.5361446142196655]