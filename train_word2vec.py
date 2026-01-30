from gensim.models import Word2Vec
import PyPDF2

PDF_PATH = "data/HRPolicyManual.pdf"
MODEL_PATH = "word2vec_hr.model"

def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def main():
    text = extract_text(PDF_PATH)
    sentences = [line.lower().split() for line in text.split(".") if line.strip()]

    print("🧠 Training Word2Vec...")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)

    model.save(MODEL_PATH)
    print(f"✅ Word2Vec model saved as {MODEL_PATH}")

if __name__ == "__main__":
    main()
