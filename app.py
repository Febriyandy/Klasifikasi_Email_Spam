from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.tokenize import word_tokenize
import io
import os
import json

app = Flask(__name__)

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

setup_nltk()

# Variabel global untuk model dan vectorizer
vectorizer = None
nb_model = None
nn_model = None
scaler = None

def preprocess_text(text):
    """
    Preprocess teks email
    """
    try:
        if pd.isna(text):
            return "kosong"
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Bersihkan teks
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenization
        tokens = text.split()
        
        # Hapus token yang terlalu pendek
        tokens = [word for word in tokens if len(word) > 1]
        
        # Gabungkan kembali
        processed = ' '.join(tokens)
        return processed if processed else "kosong"
    except Exception as e:
        print(f"Error dalam preprocessing text: {str(e)}")
        return "kosong"

def prepare_dataset(data):
    """
    Prepare dataset dan train model
    """
    global vectorizer, nb_model, nn_model, scaler
    
    try:
        # Preprocess teks
        processed_texts = data['Email Text'].apply(preprocess_text)
        
        # Vectorize teks
        vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        X = vectorizer.fit_transform(processed_texts)
        y = data['Label'].map({'Spam': 1, 'Non-Spam': 0})
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Naive Bayes
        nb_model = MultinomialNB()
        nb_model.fit(X_train, y_train)
        nb_accuracy = nb_model.score(X_test, y_test)
        
        # Scale fitur untuk Neural Network
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Neural Network dengan RBF
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        nn_model.fit(X_train_scaled, y_train)
        nn_accuracy = nn_model.score(X_test_scaled, y_test)
        
        return {
            'naive_bayes_accuracy': float(nb_accuracy),
            'neural_network_accuracy': float(nn_accuracy),
            'feature_count': int(X.shape[1]),
            'training_samples': int(X_train.shape[0]),
            'testing_samples': int(X_test.shape[0])
        }
    except Exception as e:
        raise Exception(f"Error dalam prepare dataset: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Mohon unggah file CSV'}), 400
        
        # Baca CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        data = pd.read_csv(stream, sep=';')
        
        # Verifikasi kolom yang diperlukan
        required_columns = ['Email Text', 'Label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return jsonify({
                'error': f'Kolom yang diperlukan tidak ditemukan: {", ".join(missing_columns)}'
            }), 400
        
        # Validasi Label
        valid_labels = {'Spam', 'Non-Spam'}
        invalid_labels = set(data['Label'].unique()) - valid_labels
        if invalid_labels:
            return jsonify({
                'error': f'Label tidak valid dalam dataset: {", ".join(invalid_labels)}'
            }), 400
        
        # Prepare dataset dan train models
        results = prepare_dataset(data)
        
        return jsonify({
            'message': 'Model berhasil dilatih',
            'dataset_size': len(data),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if vectorizer is None or nb_model is None or nn_model is None:
            return jsonify({
                'error': 'Model belum dilatih. Silakan upload dataset terlebih dahulu'
            }), 400
        
        # Ambil teks email
        email_text = request.json.get('email_text', '')
        
        # Preprocess teks
        processed_text = preprocess_text(email_text)
        
        # Vectorize teks
        X = vectorizer.transform([processed_text])
        
        # Prediksi dengan Naive Bayes
        nb_prediction = nb_model.predict_proba(X)[0]
        nb_is_spam = bool(nb_prediction[1] > 0.5)
        
        # Scale untuk neural network
        X_scaled = scaler.transform(X)
        nn_prediction = nn_model.predict_proba(X_scaled)[0]
        nn_is_spam = bool(nn_prediction[1] > 0.5)
        
        result = {
            'naive_bayes': {
                'spam_probability': float(nb_prediction[1]),
                'is_spam': nb_is_spam
            },
            'neural_network': {
                'spam_probability': float(nn_prediction[1]),
                'is_spam': nn_is_spam
            }
        }
        
        # Gunakan NumpyEncoder untuk mengkonversi numpy types
        return json.dumps(result, cls=NumpyEncoder)
    
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan dalam prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    setup_nltk()
    app.run(debug=True)