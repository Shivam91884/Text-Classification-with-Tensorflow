IMDB Sentiment Classification with LSTM Model

Overview
This project demonstrates a deep learning-based text classification using the IMDB movie review dataset. The dataset contains binary sentiment labels (positive/negative) associated with movie reviews. We use TensorFlow and Keras to build a Bidirectional LSTM (Long Short-Term Memory) model that classifies the sentiment of each review.

Dataset
The IMDB dataset consists of 50,000 highly polarized movie reviews, divided into 25,000 training and 25,000 test reviews. Each review is preprocessed by converting the words into integer indices.

Key Parameters:
Vocabulary Size: 10,000 (most frequent words)
Max Sequence Length: 120 (reviews are padded or truncated)
Embedding Dimension: 128
Model Architecture
The model leverages the following layers and components:

Embedding Layer: Converts word indices into dense vectors.
Bidirectional LSTM Layers: Captures context in both forward and backward directions.
Dropout Layers: Added to reduce overfitting.
Dense Layers: Fully connected layers with ReLU and Sigmoid activations for final classification.
Model Summary:
Embedding Layer: Maps word indices to dense vectors.
Bidirectional LSTM: Helps capture the sequence of words in both directions.
Dropout Layers: To mitigate overfitting.
Dense Layers: For binary classification (positive or negative sentiment).
Training
Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
The model is trained for 5 epochs with a batch size of 64, using training data and validating on the test set.

Results
After training, the model is evaluated on the test dataset:

Test Accuracy: Printed after model evaluation.
Classification Report: Includes precision, recall, and F1-score for both negative and positive classes.

Installation
Clone the repository: 
git clone https://github.com/your-username/imdb-sentiment-classification.git
cd imdb-sentiment-classification
Install dependencies:pip install -r requirements.txt

Usage
Run the script:python imdb_sentiment_classification.py
The script will:

Load and preprocess the IMDB dataset.
Build and train the LSTM-based text classification model.
Output evaluation results, including test accuracy and a classification report.
Dependencies
TensorFlow
NumPy
Scikit-learn
Install all dependencies using the requirements.txt file.
