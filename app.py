import os
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter

app = Flask(__name__)

# Configuration based on your training code
max_eng_sent_len = 22
max_fra_sent_len = 22

# Load model
model_path = os.path.join('models', 'english_to_french_translator.keras')
try:
    translator_model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize tokenizers
eng_tokenizer = None
fra_tokenizer = None

# Functions from your training code
def create_tokenizer(sentences):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    return tokenizer

def encode_sequences(tokenizer, sentences, max_sent_len):
    text_to_seq = tokenizer.texts_to_sequences(sentences)
    text_pad_seq = pad_sequences(text_to_seq, maxlen=max_sent_len, padding='post')
    return text_pad_seq

def load_or_create_tokenizers():
    global eng_tokenizer, fra_tokenizer
    
    eng_tokenizer_path = os.path.join('models', 'eng_tokenizer.pkl')
    fra_tokenizer_path = os.path.join('models', 'fra_tokenizer.pkl')
    
    # Option 1: Try to load saved tokenizers if they exist
    try:
        if os.path.exists(eng_tokenizer_path) and os.path.exists(fra_tokenizer_path):
            with open(eng_tokenizer_path, 'rb') as handle:
                eng_tokenizer = pickle.load(handle)
            with open(fra_tokenizer_path, 'rb') as handle:
                fra_tokenizer = pickle.load(handle)
            print("Loaded saved tokenizers")
            return True
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
    
    # Option 2: If we need to recreate tokenizers from your training data
    print("WARNING: Recreating tokenizers. For best results, save your tokenizers during training.")
    print("Creating empty tokenizers for now - you need to adapt this to your situation")
    
    # For a quick test, we'll create empty tokenizers
    # In your real implementation, you should:
    # 1. Save tokenizers during training using pickle (as shown in earlier examples)
    # 2. OR load the same training data here and recreate them using the same process
    eng_tokenizer = Tokenizer()
    fra_tokenizer = Tokenizer()
    
    # If you have your dataset available, you should do:
    # eng = df['English words/sentences'] 
    # fra = df['French words/sentences']
    # eng_tokenizer = create_tokenizer(eng)
    # fra_tokenizer = create_tokenizer(fra)
    
    # Instead, for now we'll just create very basic tokenizers (will not work properly)
    sample_eng = ["hello", "how are you", "thank you", "goodbye"]
    sample_fra = ["bonjour", "comment allez-vous", "merci", "au revoir"]
    
    eng_tokenizer.fit_on_texts(sample_eng)
    fra_tokenizer.fit_on_texts(sample_fra)
    
    print("Warning: Using placeholder tokenizers - translation will not work correctly")
    return False

def translate_text(text):
    # Process input text - single sentence translation
    input_sequence = eng_tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_sequence, maxlen=max_eng_sent_len, padding='post')
    
    # Make prediction
    prediction = translator_model.predict(input_padded)
    
    # Decode the prediction
    output_sequence = np.argmax(prediction, axis=2)[0]
    
    # Convert indices to words
    word_index = {v: k for k, v in fra_tokenizer.word_index.items()}
    
    output_words = []
    for idx in output_sequence:
        if idx > 0 and idx in word_index:  # Skip padding (0)
            word = word_index[idx]
            output_words.append(word)
    
    return ' '.join(output_words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        english_text = request.form.get('english_text', '')
        
        if not english_text:
            return jsonify({'error': 'Please enter English text'}), 400
        
        # Translate the text
        french_text = translate_text(english_text)
        
        return jsonify({'translation': french_text})
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Translation error: {str(e)}'}), 500

if __name__ == '__main__':
    # Make sure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load or create tokenizers
    load_or_create_tokenizers()
    
    app.run(debug=True)