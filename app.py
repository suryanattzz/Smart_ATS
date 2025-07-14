# from flask import Flask, render_template, request
# import os
# import fitz  # PyMuPDF for PDF reading
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'pdf'}

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Ensure upload folder exists
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text()
#     return text

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')

# @app.route('/results', methods=['POST'])
# def results():
#     job_description = request.form['job_description']
#     uploaded_files = request.files.getlist("resumes")
    
#     candidates = []
#     for file in uploaded_files:
#         if file and allowed_file(file.filename):
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(file_path)
#             resume_text = extract_text_from_pdf(file_path)
#             print(resume_text)
#             candidates.append((file.filename, resume_text))

#     # NLP: TF-IDF and Cosine Similarity
#     texts = [job_description] + [c[1] for c in candidates]
#     vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

#     ranked_candidates = sorted(zip([c[0] for c in candidates], scores), key=lambda x: x[1], reverse=True)

#     return render_template('results.html', candidates=ranked_candidates)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF for PDF reading
from sentence_transformers import SentenceTransformer, util

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the BERT-based model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast, and accurate

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    job_description = request.form['job_description']
    uploaded_files = request.files.getlist("resumes")

    candidates = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            resume_text = extract_text_from_pdf(file_path)
            candidates.append((file.filename, resume_text))

    # Encode job description using BERT
    jd_embedding = model.encode(job_description, convert_to_tensor=True)

    ranked_candidates = []
    for filename, resume_text in candidates:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
        ranked_candidates.append((filename, similarity))

    # Sort by descending similarity
    ranked_candidates.sort(key=lambda x: x[1], reverse=True)

    return render_template('results.html', candidates=ranked_candidates)

if __name__ == '__main__':
    app.run(debug=True)
