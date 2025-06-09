from flask import Flask, request, jsonify, render_template
import spacy
import requests
from flask_cors import CORS

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # Bolehkan akses dari frontend

# Load model bahasa Indonesia
nlp = spacy.load("id_core_news_sm")

# Kamus genre
genre_openlibrary = {
    "fantasi": "fantasy",
    "petualangan": "adventure",
    "romantis": "romance",
    "ilmu_pengetahuan": "science_fiction",
    "thriller": "thriller",
    "sejarah": "history",
    "horor": "horror",
    "olahraga": "sports",
    "bisnis": "business",
    "motivasi": "self_help",
    "agama": "religion",
    "psikologi": "psychology",
    "seni": "art",
    "musik": "music",
    "kesehatan": "health",
}

# Fungsi NLP
def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def ekstrak_genre(teks):
    lemmatized = lemmatize(teks.lower())
    ditemukan = []
    for kata in lemmatized:
        for kunci in genre_openlibrary:
            if kunci in kata and genre_openlibrary[kunci] not in ditemukan:
                ditemukan.append(genre_openlibrary[kunci])
    return ditemukan

# Ambil buku dari Open Library
def cari_buku_dari_openlibrary(genre_en):
    url = f"https://openlibrary.org/subjects/{genre_en}.json?limit=5"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return []
        data = response.json()
        buku = []
        for work in data.get("works", []):
            judul = work.get("title")
            penulis = ", ".join([a.get("name") for a in work.get("authors", [])]) if work.get("authors") else "Unknown"
            buku.append({"judul": judul, "penulis": penulis})
        return buku
    except:
        return []

# ✅ Route untuk menampilkan index.html
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Endpoint API utama
@app.route("/saran-buku", methods=["POST"])
def saran_buku():
    data = request.get_json()
    user_input = data.get("minat", "")
    
    if not user_input.strip():
        return jsonify({"error": "Input kosong"}), 400

    genres = ekstrak_genre(user_input)
    if not genres:
        return jsonify({"message": "Maaf, kami belum bisa memberi saran.", "buku": []})

    hasil = []
    for genre in genres:
        hasil.extend(cari_buku_dari_openlibrary(genre))

    if not hasil:
        return jsonify({"message": "Maaf, kami belum bisa memberi saran.", "buku": []})

    return jsonify({"message": "Berikut saran buku untukmu", "buku": hasil})

# Run app
if __name__ == "__main__":
    app.run(debug=True)
