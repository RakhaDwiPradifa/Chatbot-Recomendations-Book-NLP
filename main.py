from flask import Flask, request, jsonify, render_template
import spacy
import requests
from flask_cors import CORS
import random
from collections import Counter
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Load model bahasa Indonesia
nlp = spacy.load("id_core_news_sm")

# Inisialisasi Sastrawi stemmer dan stopword remover
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Kamus genre yang diperluas
genre_openlibrary = {
    "fantasi": "fantasy",
    "petualangan": "adventure",
    "romantis": "romance",
    "cinta": "romance",
    "ilmu_pengetahuan": "science_fiction",
    "sains": "science_fiction",
    "teknologi": "science_fiction",
    "thriller": "thriller",
    "misteri": "mystery",
    "sejarah": "history",
    "horor": "horror",
    "olahraga": "sports",
    "bisnis": "business",
    "motivasi": "self_help",
    "pengembangan_diri": "self_help",
    "agama": "religion",
    "spiritual": "religion",
    "psikologi": "psychology",
    "seni": "art",
    "musik": "music",
    "kesehatan": "health",
    "pendidikan": "education",
    "anak": "children",
    "masakan": "cooking",
    "travel": "travel",
    "filosofi": "philosophy"
}

# Template respon dinamis
greeting_templates = [
    "Berdasarkan minatmu tentang {}, berikut beberapa buku yang mungkin kamu suka:",
    "Aha! Kamu tertarik dengan {}? Aku punya beberapa rekomendasi bagus nih:",
    "Suka {}? Keren! Coba cek buku-buku menarik ini:",
    "Untuk penggemar {}, aku menemukan beberapa buku yang wajib dibaca:",
    "Sepertinya kamu suka {}! Ini beberapa rekomendasi yang cocok untukmu:"
]

no_results_templates = [
    "Hmm, untuk topik {} sepertinya aku belum menemukan buku yang cocok. Coba ceritakan minatmu yang lain?",
    "Maaf ya, aku masih kesulitan menemukan buku tentang {}. Mungkin bisa coba kata kunci lain?",
    "Untuk {} aku belum punya rekomendasi yang pas. Bagaimana kalau kita coba topik lain?",
    "Sepertinya belum ada buku {} yang cocok di database. Mau coba mencari genre lain?"
]

class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("id_core_news_sm")
        self.stemmer = stemmer
        self.stopword_remover = stopword_remover
        
    def tokenize(self, text):
        """Tokenisasi teks menggunakan spaCy"""
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            tokens.append({
                'text': token.text,
                'pos': token.pos_,  # Part of speech
                'is_stop': token.is_stop  # Whether it's a stopword
            })
        return tokens
    
    def remove_stopwords(self, text):
        """Menghapus stopwords menggunakan Sastrawi"""
        return self.stopword_remover.remove(text)
    
    def stem_text(self, text):
        """Melakukan stemming menggunakan Sastrawi"""
        return self.stemmer.stem(text)
    
    def lemmatize(self, text):
        """Melakukan lemmatization menggunakan spaCy"""
        doc = self.nlp(text)
        lemmas = []
        for token in doc:
            lemmas.append({
                'original': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_
            })
        return lemmas
    
    def detect_phrases(self, text):
        """Deteksi frasa menggunakan spaCy"""
        doc = self.nlp(text)
        phrases = []
        
        # Deteksi noun phrases
        for chunk in doc.noun_chunks:
            phrases.append({
                'text': chunk.text,
                'root': chunk.root.text,
                'type': 'noun_phrase'
            })
        
        # Deteksi verb phrases
        for token in doc:
            if token.pos_ == "VERB":
                phrase = token.text
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        phrase += " " + child.text
                phrases.append({
                    'text': phrase,
                    'root': token.text,
                    'type': 'verb_phrase'
                })
        
        return phrases
    
    def parse_dependencies(self, text):
        """Analisis dependensi sintaksis"""
        doc = self.nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'token': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text
            })
        
        return dependencies
    
    def process_text(self, text):
        """Proses teks lengkap dengan semua tahapan"""
        # 1. Tokenisasi
        tokens = self.tokenize(text)
        
        # 2. Hapus stopwords
        text_no_stop = self.remove_stopwords(text)
        
        # 3. Stemming
        stemmed_text = self.stem_text(text_no_stop)
        
        # 4. Lemmatization
        lemmas = self.lemmatize(text)
        
        # 5. Deteksi frasa
        phrases = self.detect_phrases(text)
        
        # 6. Parsing dependensi
        dependencies = self.parse_dependencies(text)
        
        # 7. Proses descriptions
        process_descriptions = {
            'tokenization': 'Memecah teks menjadi token-token individual',
            'stopword_removal': 'Menghapus kata-kata umum yang tidak memiliki makna spesifik',
            'stemming': 'Mengubah kata berimbuhan menjadi kata dasar',
            'lemmatization': 'Mengubah kata menjadi bentuk dasar dengan mempertimbangkan konteks',
            'phrase_detection': 'Mendeteksi kelompok kata yang membentuk frasa bermakna',
            'dependency_parsing': 'Menganalisis hubungan gramatikal antar kata'
        }
        
        return {
            'original': text,
            'tokens': tokens,
            'no_stopwords': text_no_stop,
            'stemmed': stemmed_text,
            'lemmas': lemmas,
            'phrases': phrases,
            'dependencies': dependencies,
            'process_descriptions': process_descriptions
        }

# Inisialisasi text processor
text_processor = TextProcessor()

def extract_keywords(text):
    """Ekstrak kata kunci penting dari input pengguna dengan NLP yang lebih kompleks"""
    # Proses teks lengkap
    processed = text_processor.process_text(text)
    
    # Gunakan hasil stemming dan remove stopwords
    clean_text = processed['stemmed']
    
    # Analisis dengan spaCy
    doc = nlp(clean_text)
    
    keywords = []
    # Tambahkan kata-kata penting berdasarkan POS tag
    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop:
            keywords.append(token.lemma_)
    
    # Tambahkan frasa penting
    for phrase in processed['phrases']:
        if len(phrase['text'].split()) > 1:  # Hanya frasa dengan >1 kata
            keywords.append(phrase['text'])
    
    # Hapus duplikat dan kembalikan
    return list(set(keywords))

def analyze_sentiment(text):
    """Analisis sentimen yang lebih kompleks dengan mempertimbangkan struktur kalimat"""
    # Proses teks untuk mendapatkan dependensi dan struktur
    processed = text_processor.process_text(text)
    
    positive_words = {'suka', 'senang', 'bagus', 'baik', 'menarik', 'indah', 'keren', 'hebat', 'mantap', 'asyik'}
    negative_words = {'tidak', 'benci', 'buruk', 'jelek', 'bosan', 'malas', 'payah', 'gagal'}
    intensifiers = {'sangat', 'sekali', 'banget', 'terlalu', 'amat'}
    
    sentiment_score = 0
    
    # Analisis berdasarkan dependensi
    for dep in processed['dependencies']:
        word = dep['token'].lower()
        
        # Cek sentimen dasar
        if word in positive_words:
            score = 1
            # Cek negasi
            if any(t['token'].lower() in ['tidak', 'bukan', 'jangan'] for t in processed['dependencies'] 
                  if t['head'] == dep['token']):
                score *= -1
            # Cek intensifier
            if any(t['token'].lower() in intensifiers for t in processed['dependencies'] 
                  if t['head'] == dep['token']):
                score *= 2
            sentiment_score += score
            
        elif word in negative_words:
            score = -1
            # Cek intensifier
            if any(t['token'].lower() in intensifiers for t in processed['dependencies'] 
                  if t['head'] == dep['token']):
                score *= 2
            sentiment_score += score
    
    return sentiment_score

def get_dynamic_response(genres, sentiment_score):
    """Menghasilkan respons yang dinamis berdasarkan genre dan sentimen"""
    if not genres:
        return "Hmm, ceritakan lebih detail tentang minatmu ya? Aku ingin memberikan rekomendasi yang benar-benar sesuai!"
    
    genre_str = " dan ".join(genres)
    if sentiment_score > 0:
        return random.choice(greeting_templates).format(genre_str)
    elif sentiment_score < 0:
        return f"Wah, sepertinya kamu kurang suka dengan {genre_str}? Tapi tenang, aku punya beberapa rekomendasi yang mungkin bisa mengubah pandanganmu:"
    else:
        return random.choice(greeting_templates).format(genre_str)

def score_book_relevance(book, keywords):
    """Menghitung skor relevansi buku berdasarkan kata kunci"""
    score = 0
    title_words = set(re.findall(r'\w+', book['judul'].lower()))
    
    for keyword in keywords:
        if keyword in title_words:
            score += 2
        if any(keyword in author.lower() for author in book['penulis'].split(', ')):
            score += 1
    
    return score

def ekstrak_genre(teks):
    """Ekstrak genre dengan penanganan lebih cerdas menggunakan NLP"""
    # Proses teks lengkap
    processed = text_processor.process_text(teks)
    
    ditemukan = []
    
    # Gunakan hasil stemming untuk pencocokan genre
    stemmed_text = processed['stemmed']
    
    # Cek kata dasar setelah stemming
    for kata in stemmed_text.split():
        for kunci in genre_openlibrary:
            if kunci in kata and genre_openlibrary[kunci] not in ditemukan:
                ditemukan.append(genre_openlibrary[kunci])
    
    # Cek frasa untuk konteks yang lebih spesifik
    for phrase in processed['phrases']:
        phrase_text = phrase['text'].lower()
        
        # Deteksi genre berdasarkan frasa
        if any(word in phrase_text for word in ['belajar', 'pelajaran', 'kuliah', 'pendidikan']):
            if 'education' not in ditemukan:
                ditemukan.append('education')
        if any(word in phrase_text for word in ['masak', 'resep', 'makanan', 'kuliner']):
            if 'cooking' not in ditemukan:
                ditemukan.append('cooking')
        if any(word in phrase_text for word in ['jalan-jalan', 'traveling', 'wisata', 'perjalanan']):
            if 'travel' not in ditemukan:
                ditemukan.append('travel')
    
    # Gunakan dependensi untuk konteks tambahan
    deps = processed['dependencies']
    for dep in deps:
        if dep['pos'] == 'NOUN':  # Fokus pada kata benda
            word = dep['token'].lower()
            # Tambahkan pengecekan khusus berdasarkan konteks
            if word in ['novel', 'cerita']:
                head = dep['head'].lower()
                if head in ['romantis', 'cinta']:
                    ditemukan.append('romance')
                elif head in ['seram', 'horor']:
                    ditemukan.append('horror')
    
    return list(set(ditemukan))  # Hapus duplikat

def cari_buku_dari_openlibrary(genre_en):
    """Pencarian buku yang ditingkatkan"""
    url = f"https://openlibrary.org/subjects/{genre_en}.json?limit=8"  # Increased limit
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return []
        data = response.json()
        buku = []
        for work in data.get("works", []):
            judul = work.get("title")
            penulis = ", ".join([a.get("name") for a in work.get("authors", [])]) if work.get("authors") else "Unknown"
            
            # Tambahkan informasi tambahan jika tersedia
            cover_id = work.get("cover_id")
            cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None
            
            buku.append({
                "judul": judul,
                "penulis": penulis,
                "cover_url": cover_url,
                "first_publish_year": work.get("first_publish_year"),
                "subject": work.get("subject", [])[:3]  # Ambil 3 subjek pertama
            })
        return buku
    except Exception as e:
        print(f"Error fetching books: {str(e)}")
        return []

def get_relevant_facts(processed_text, genres):
    """Menghasilkan fakta-fakta menarik berdasarkan input dan genre yang terdeteksi"""
    facts = []
    
    # Fakta tentang analisis teks
    word_count = len(processed_text['tokens'])
    facts.append(f"📝 Input kamu mengandung {word_count} kata")
    
    # Fakta tentang frasa
    phrases = processed_text['phrases']
    if phrases:
        noun_phrases = [p['text'] for p in phrases if p['type'] == 'noun_phrase']
        if noun_phrases:
            facts.append(f"🔍 Frasa utama yang terdeteksi: {', '.join(noun_phrases[:2])}")
    
    # Fakta tentang genre
    if genres:
        genre_facts = {
            'romance': 'Genre roman adalah salah satu genre paling populer di dunia literatur',
            'fantasy': 'Buku fantasi membantu mengembangkan kreativitas dan imajinasi',
            'science_fiction': 'Fiksi ilmiah sering menginspirasi penemuan teknologi nyata',
            'mystery': 'Genre misteri meningkatkan kemampuan berpikir analitis',
            'history': 'Membaca buku sejarah membantu kita belajar dari masa lalu',
            'self_help': 'Buku pengembangan diri dapat meningkatkan kualitas hidup',
            'psychology': 'Psikologi membantu memahami perilaku manusia',
            'education': 'Pendidikan adalah kunci untuk membuka pintu kesuksesan'
        }
        for genre in genres:
            if genre in genre_facts:
                facts.append(f"💡 {genre_facts[genre]}")
    
    return facts[:3]  # Batasi menjadi 3 fakta saja

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/saran-buku", methods=["POST"])
def saran_buku():
    data = request.get_json()
    user_input = data.get("minat", "").strip()
    
    if not user_input:
        return jsonify({
            "message": "Ups! Kamu belum menulis apa-apa nih. Yuk, ceritakan minatmu!",
            "buku": []
        }), 400

    # Proses teks lengkap
    processed = text_processor.process_text(user_input)
    
    # Analisis input pengguna
    keywords = extract_keywords(user_input)
    sentiment = analyze_sentiment(user_input)
    genres = ekstrak_genre(user_input)
    
    # Dapatkan fakta-fakta menarik
    facts = get_relevant_facts(processed, genres)

    # Persiapkan hasil analisis NLP
    nlp_analysis = {
        "tokens": [{'text': t['text'], 'pos': t['pos'], 'is_stop': t['is_stop']} for t in processed['tokens']],
        "cleaned_text": processed['no_stopwords'],
        "stemmed_words": processed['stemmed'].split(),
        "lemmas": [{'original': l['original'], 'lemma': l['lemma'], 'pos': l['pos']} for l in processed['lemmas']],
        "detected_phrases": [p['text'] for p in processed['phrases']],
        "dependencies": processed['dependencies'],
        "process_descriptions": processed['process_descriptions'],
        "facts": facts
    }

    if not genres:
        return jsonify({
            "message": "Hmm, aku masih belajar nih. Bisa ceritakan lebih spesifik tentang minatmu? Misalnya: 'Aku suka buku fantasi' atau 'Aku tertarik dengan sejarah'",
            "buku": [],
            "nlp_analysis": nlp_analysis
        })

    # Kumpulkan buku dari semua genre
    semua_buku = []
    for genre in genres:
        buku_genre = cari_buku_dari_openlibrary(genre)
        for buku in buku_genre:
            buku['relevance_score'] = score_book_relevance(buku, keywords)
            semua_buku.append(buku)

    # Sortir berdasarkan skor relevansi
    semua_buku.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Ambil 5 buku terbaik
    buku_terpilih = semua_buku[:5]

    if not buku_terpilih:
        response_message = random.choice(no_results_templates).format("/".join(genres))
    else:
        response_message = get_dynamic_response(genres, sentiment)

    return jsonify({
        "message": response_message,
        "buku": buku_terpilih,
        "genres_detected": genres,
        "nlp_analysis": nlp_analysis
    })

if __name__ == "__main__":
    app.run(debug=True)
