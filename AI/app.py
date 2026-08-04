import re
from flask import Flask, request, jsonify, send_from_directory
import os
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Ensure required NLTK resources are available (download only if missing)
def _ensure_nltk_resource(res_name: str, res_path: str):
    try:
        nltk.data.find(res_path)
    except LookupError:
        nltk.download(res_name)

_ensure_nltk_resource('stopwords', 'corpora/stopwords')
_ensure_nltk_resource('wordnet', 'corpora/wordnet')
_ensure_nltk_resource('punkt', 'tokenizers/punkt')

app = Flask(__name__)
CORS(app)  # Allows your front-end to communicate with this backend

wordnet = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Serve index.html at root if it exists, and provide a simple health check
@app.route('/', methods=['GET'])
def index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(base_dir, 'index.html')
    return jsonify({"message": "API is running"}), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    paragraph = data.get('text', '')
    
    # 1. Tokenize into sentences
    sentences = nltk.sent_tokenize(paragraph)
    corpus = []
    
    # 2. Text Cleaning & Preprocessing loop
    for i in range(len(sentences)):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower().split()
        review = [wordnet.lemmatize(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        if review.strip():  # Skip empty lines
            corpus.append(review)
            
    if not corpus:
        return jsonify({"error": "No valid text to process"}), 400

    # 3. Bag of Words Vectorization
    cv1 = CountVectorizer()
    X1 = cv1.fit_transform(corpus).toarray().tolist()  # Convert to standard Python list for JSON
    bow_vocab = cv1.get_feature_names_out().tolist()

    # 4. TF-IDF Vectorization
    cv = TfidfVectorizer()
    X = cv.fit_transform(corpus).toarray().tolist()
    tfidf_vocab = cv.get_feature_names_out().tolist()

    # Send the numerical calculations back to the UI
    return jsonify({
        "corpus": corpus,
        "bow_matrix": X1,
        "bow_vocab": bow_vocab,
        "tfidf_matrix": X,
        "tfidf_vocab": tfidf_vocab
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
