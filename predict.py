from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


app = Flask(__name__)
app.secret_key = 'my_secretKey' 

# Load your models
stacking_model = joblib.load("Models/stacking_model.pkl")
nb_model = joblib.load("Models/naive_bayes_model.pkl")
dt_model = joblib.load('Models/decision_tree_model.pkl')
knn_model = joblib.load('Models/knn_model.pkl')
vectorizer = joblib.load("Models/tfidf_vectorizer.pkl")  # Ensure you have saved this if used
# Load the model from the file
bert_Model = tf.keras.models.load_model('Models/bert_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

# Initialize stopwords and lemmatizer
stop_words = stopwords.words("English")
wn.ensure_loaded()
wn1 = WordNetLemmatizer()
def clean_text(text):
    """
    Function to clean text by lowercasing, removing special characters and digits,
    and removing stop words.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

def tokenise_and_lemmatize(text): 
    """
    Tokenize and Lemmatize text 
    """
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [wn1.lemmatize(token) for token in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

def preprocess_text(text):
    text = clean_text(text)
    text = tokenise_and_lemmatize(text)
    return text

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        preprocessed_text = preprocess_text(text)
        X_test_tfidf = vectorizer.transform([preprocessed_text])
        
        bertText = [text]
        bert_pred = bert_Model.predict(bertText)
        bert_category = 1 if bert_pred >= 0.5 else 0

        nb_pred = nb_model.predict(X_test_tfidf)[0]
        dt_pred = dt_model.predict(X_test_tfidf)[0]
        knn_pred = knn_model.predict(X_test_tfidf)[0]

        stacked_input = [[nb_pred, dt_pred, knn_pred, bert_category]]
        stacking_prediction = stacking_model.predict(stacked_input)[0]

        final_message = "Fake" if stacking_prediction == 1 else "Real"
        
        predictions = {
            "Naive Bayes": "Fake" if nb_pred == 1 else "Real",
            "Decision Tree": "Fake" if dt_pred == 1 else "Real",
            "KNN": "Fake" if knn_pred == 1 else "Real",
            "BERT": "Fake" if bert_category == 1 else "Real",
            "Stacked Model": final_message
        }

        session['predictions'] = predictions  # Store predictions in session
        return render_template('index.html', results=final_message)

@app.route('/view_report')
def view_report():
    predictions = session.get('predictions', {})
    return render_template('view_report.html', predictions=predictions)

@app.route('/view_models')
def view_models():
    return render_template('view_models.html')


if __name__ == '__main__':
    app.run(debug=True)
