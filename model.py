import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("balanced_url_dataset.csv")

# Drop NaN values
data = data.dropna()

# Drop duplicates
data = data.drop_duplicates()

# Convert URLs to lowercase
data['url'] = data['url'].str.lower()

# Encode labels: phishing = 0, legitimate = 1
data['label'] = data['type'].map({'legitimate': 1, 'phishing': 0})

# Keep only necessary columns
data = data[['url', 'label']]

# print(data.head())
print(data['label'].value_counts())

from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from keras.preprocessing.sequence import pad_sequences

# Convert URLs to lowercase
data['url'] = data['url'].str.lower()

# Tokenizer setup (we can limit vocab size if dataset is huge)
tokenizer = Tokenizer(char_level=True)  # character-level works best for URLs
tokenizer.fit_on_texts(data['url'])

# Convert URLs to sequences of numbers
sequences = tokenizer.texts_to_sequences(data['url'])

# Pad sequences (make all URLs same length)
maxlen = 200  # trim/pad URLs to 200 characters
X = pad_sequences(sequences, maxlen=maxlen)

# Labels
y = data['label'].values

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout #type: ignore

vocab_size = len(tokenizer.word_index) + 1  # total unique characters + 1 for padding
embedding_dim = 64  # size of embedding vectors

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=X.shape[1]))
model.add(LSTM(128, return_sequences=False))  # 128 LSTM units
model.add(Dropout(0.5))  # helps prevent overfitting
model.add(Dense(1, activation='sigmoid'))  # binary classification

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Training parameters
batch_size = 64
epochs = 10  # you can increase later for better accuracy

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)