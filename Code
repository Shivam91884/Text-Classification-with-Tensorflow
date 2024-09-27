import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.datasets import imdb
from sklearn.metrics import classification_report
# Step 1: Load and preprocess the IMDB dataset
# Load the IMDB dataset (binary sentiment classification)
vocab_size = 10000  # Use the top 10,000 most frequent words
max_length = 120    # Maximum sequence length (truncated/padded)
embedding_dim = 128 # Embedding dimension for word vectors
oov_tok = "<OOV>"  # Token for out-of-vocabulary words

# Load dataset from Keras, setting a vocabulary limit
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Step 2: Padding the sequences to have consistent input length
# Pad training and testing data
x_train_padded = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test_padded = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
# Step 3: Define the text classification model using TensorFlow
# Initialize the model
model = Sequential()

# Embedding Layer: Converts word indices to dense vectors of a fixed size
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

# LSTM Layer: Adds Long Short-Term Memory (LSTM) network to capture sequential data
model.add(Bidirectional(LSTM(64, return_sequences=True)))  # Bidirectional LSTM for capturing context in both directions
model.add(Dropout(0.5))  # Regularization to prevent overfitting
model.add(LSTM(32))      # Another LSTM layer

# Dense Layers: Fully connected layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Additional dropout layer
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification

# Step 4: Compile the model
# Adam optimizer and binary crossentropy loss for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Step 5: Train the model
# Train the model with training data and validate on test data
history = model.fit(x_train_padded, y_train, epochs=5, batch_size=64, validation_data=(x_test_padded, y_test), verbose=1)
# Step 6: Evaluate the model
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test_padded, y_test, verbose=2)
print('Test Accuracy:', test_acc)
# Step 7: Generate classification report
y_pred = (model.predict(x_test_padded) > 0.5).astype("int32")  # Predict classes using a threshold of 0.5
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
