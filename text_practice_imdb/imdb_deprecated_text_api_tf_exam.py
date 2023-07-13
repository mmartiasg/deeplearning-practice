# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/utils/pad_sequences
# Deprecated since 2.9.0
# It says I should use TextVectorizer but this last one does not allow to save the model in h5 format!
# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/preprocessing/text/Tokenizer

import tensorflow as tf
import numpy as np
import math

# limit ram
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

# Optimizations
# GPU
tf.config.optimizer.set_jit(True)
# CPU?
tf.function(jit_compile=True)
# Mixed precision
# not compatible with trained weights of the efficient net
tf.keras.mixed_precision.set_global_policy('mixed_float16')

(X, Y), (X_Test, Y_Test) = tf.keras.datasets.imdb.load_data(num_words=30000)


# Reconvert to text
def index_to_text_sentences(X):
    word_to_index = tf.keras.datasets.imdb.get_word_index()
    index_to_word = dict([(index, word) for word, index in word_to_index.items()])
    X_Text = []
    for sentence_index in X:
        text = " ".join([index_to_word[token] for token in sentence_index])
        X_Text.append(text)

    return X_Text


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

X_Train = index_to_text_sentences(X_Train)
X_Val = index_to_text_sentences(X_Val)
X_Test = index_to_text_sentences(X_Test)

VOC_SIZE = 10000
OUTPUT_MAX_LENGTH = 1000
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=VOC_SIZE,
    oov_token="[OOV]",
    split=" "
)

tokenizer.fit_on_texts(X_Train)


# Preprocess Text
def preprocess_text(text):
    tokenized_text = tokenizer.texts_to_sequences([text.split(" ")])
    return tf.keras.preprocessing.sequence.pad_sequences(tokenized_text, maxlen=OUTPUT_MAX_LENGTH, padding="post", truncating="post")


def tokenize_padding_text(corpus):
    tokenized = []
    for sentence in corpus:
        tokenized.append(preprocess_text(sentence)[0])

    return tokenized


X_Train = np.array(tokenize_padding_text(X_Train))
X_Val = np.array(tokenize_padding_text(X_Val))
X_Test = np.array(tokenize_padding_text(X_Test))

Y_Train = Y_Train.reshape([-1, 1])
Y_Val = Y_Val.reshape([-1, 1])
Y_Test = Y_Test.reshape([-1, 1])

print("Train shape")
print(X_Train[0].shape)
print(X_Train.shape)
print(Y_Train[0].shape)
print(Y_Train.shape)

print("VAL shape")
print(X_Val[0].shape)
print(X_Val.shape)
print(Y_Val[0].shape)
print(Y_Val.shape)

print("Test shape")
print(X_Test[0].shape)
print(X_Test.shape)
print(Y_Test[0].shape)
print(Y_Test.shape)

train_data = tf.data.Dataset.from_tensor_slices((X_Train, Y_Train)).batch(512).prefetch(tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((X_Val, Y_Val)).batch(128).prefetch(tf.data.AUTOTUNE)
test_data = tf.data.Dataset.from_tensor_slices((X_Test, Y_Test)).batch(128).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(OUTPUT_MAX_LENGTH,), dtype="int64"),
    tf.keras.layers.Embedding(input_dim=VOC_SIZE, output_dim=64, mask_zero=True),
    # Recurrent droput inproves in val performane by >2% from 84% to 86.98%
    # [0.3210374414920807, 0.8705199956893921]
    tf.keras.layers.LSTM(32, recurrent_dropout=0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")])

model.compile(loss="binary_crossentropy", metrics=["accuracy"],
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3))

print(model.summary())


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Define callbacks
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint", save_only_best=True,
                                                     save_weights_only=True, verbose=1)

DEVICE = "GPU"

# Train
with tf.device(f"/{DEVICE}:0"):
    model.fit(train_data,
              validation_data=val_data,
              epochs=500,
              callbacks=[early_stop_callback, save_best_model, reduce_lr])

    print(f"Evaluation metrics last model: {model.evaluate(test_data)}")

    model.load_weights("best_model")
    print(f"Evaluation metrics best model's weights: {model.evaluate(test_data)}")

    model.save(filepath="imdb-model.h5", save_format="h5")
