import tensorflow as tf
import numpy as np
import math
import re

"""
Reuters is a benchmark dataset for document classification.
46 classes!
"""
(X, Y), (X_Test, Y_Test) = tf.keras.datasets.reuters.load_data(num_words=30000)

# limit ram to run multiple algorithms at once
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
# Optimizations
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Transform array of indexes into full text sentences.
reverse_dictionary = dict([(index, word) for word, index in tf.keras.datasets.reuters.get_word_index().items()])

X = np.array([" ".join([reverse_dictionary[token] for token in sentence_index]) for sentence_index in X])
X_Test = np.array([" ".join([reverse_dictionary[token] for token in sentence_index]) for sentence_index in X_Test])

X = np.array([re.sub(r'[0-9]+', '[NUMBER]', x) for x in X])

# oov_token, padding, truncationg model will be provided
OOV_TOKEN = "[UNK]"
PADDING = "post"
TRUNCATING = "post"
VOCABULARY_SIZE = 20000
CLASSES = 46

vectorizer = tf.keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE,
                                                   oov_token=OOV_TOKEN, char_level=False, split=" ")
padding = tf.keras.preprocessing.sequence.pad_sequences

# Split dataset
N = X.shape[0]

n_train = math.ceil(N * 0.7)

np.random.RandomState(42).shuffle(X)
np.random.RandomState(42).shuffle(Y)

X_Train = X[:n_train]
Y_Train = Y[:n_train]

X_Val = X[n_train:]
Y_Val = Y[n_train:]


# Make dataset
def transform_text_to_index(batch_texts, batch_labels):
    vectorized_batch = vectorizer.texts_to_sequences(batch_texts)
    padded_vectorized_batch = padding(vectorized_batch, maxlen=600, padding=PADDING, truncating=TRUNCATING)
    return padded_vectorized_batch, batch_labels


# do not forget to fit the vectorizer first!!
vectorizer.fit_on_texts(X_Train)
#####################################

X_Train, Y_Train = transform_text_to_index(X_Train, Y_Train)
X_Val, Y_Val = transform_text_to_index(X_Val, Y_Val)

print(X_Train[0][:5])
print(X_Train[0].shape)
print(X_Train.shape)
print(Y_Train.shape)

train_dataset = (tf.data.Dataset
                   .from_tensor_slices((X_Train, Y_Train))
                   .shuffle(256)
                   .batch(256)
                   .prefetch(tf.data.AUTOTUNE)
                   # .map(transform_text_to_index)
                   )
val_dataset = (tf.data.Dataset
                 .from_tensor_slices((X_Val, Y_Val))
                 .batch(64)
                 .prefetch(tf.data.AUTOTUNE)
                 # .map(transform_text_to_index)
                 )
test_dataset = (tf.data.Dataset
                .from_tensor_slices((X_Test, Y_Test))
                .batch(64)
                .prefetch(tf.data.AUTOTUNE)
                # .map(transform_text_to_index)
                )

# model
# model = tf.keras.Sequential([
#     tf.keras.Input(shape=(600,)),
#     tf.keras.layers.Embedding(input_dim=VOCABULARY_SIZE, output_dim=64, mask_zero=True),
#     # last best
#     # 1 lstm 32 units
#     # - 19s 781ms/step - loss: 1.5890 - sparse_top_k_categorical_accuracy: 0.8828 - val_loss: 1.7186 - val_sparse_top_k_categorical_accuracy: 0.8664
#     tf.keras.layers.LSTM(units=32, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)),
#     tf.keras.layers.LSTM(units=32, return_sequences=False, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)),
#     tf.keras.layers.Dense(units=CLASSES, activation="softmax")
# ])

# model sep 1D convolutions
model = tf.keras.Sequential([
    tf.keras.Input(shape=(600,)),
    tf.keras.layers.Embedding(input_dim=VOCABULARY_SIZE, output_dim=64, mask_zero=True),
    tf.keras.layers.SeparableConvolution1D(filters=16, kernel_size=5,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)),
    # tf.keras.layers.MaxPooling1D(strides=2),
    # tf.keras.layers.SeparableConvolution1D(filters=16, kernel_size=5,
    #                                        kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)),
    # tf.keras.layers.MaxPooling1D(strides=2),
    # tf.keras.layers.SeparableConvolution1D(filters=16, kernel_size=5,
    #                                        kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(units=CLASSES, activation="softmax")
])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temporal_checkpoint",
                                                      save_weights_only=True,
                                                      save_only_best=True,
                                                      verbose=1)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)],
              jit_compile=True)

# Fit
model.fit(train_dataset, validation_data=val_dataset, epochs=500, callbacks=[early_stop, model_checkpoint])
model.load_weights("temporal_checkpoint")
model.save(filepath="reuters.h5", save_format="h5")

print(f"model evaluation: {model.evaluate(test_dataset)}")
