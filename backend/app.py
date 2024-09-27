from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from werkzeug.utils import secure_filename
import re
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel, CLIPModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import BeitImageProcessor, BeitForImageClassification
from torchvision import transforms
import numpy as np
from torchvision.models import swin_t, Swin_T_Weights
from transformers import AutoImageProcessor, SwinForImageClassification
from transformers import AutoImageProcessor, ViTMAEModel
import matplotlib.pyplot as plt

os.environ['OPENAI_API_KEY'] = ''
app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
    })
app.config['CORS_HEADERS'] = 'Content-Type'

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads/')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

data = {
    "Tahun": [2015, 2016, 2017, 2018, 2019],
    "Penjualan (Juta Rp)": [100, 120, 150, 180, 200],
    "Biaya Iklan (Juta Rp)": [20, 25, 30, 35, 40]
}

df = pd.DataFrame(data)

x = df[["Biaya Iklan (Juta Rp)"]]
y = df["Penjualan (Juta Rp)"]

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(x, y)
coefficient = df["Penjualan (Juta Rp)"].corr(df["Biaya Iklan (Juta Rp)"])
prediction = model.predict([[45]])[0]
def apply_canny_edge_detection(image, low_threshold=100, high_threshold=200):
    """
    Apply Canny edge detection to an input image.

    Args:
    image (numpy.ndarray): Input image (color or grayscale)
    low_threshold (int): Lower threshold for edge detection (default: 100)
    high_threshold (int): Upper threshold for edge detection (default: 200)

    Returns:
    numpy.ndarray: Image with detected edges
    """
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    return edges


def apply_edge_detection(image, method="canny", canny_low=100, canny_high=200, sobel_ksize=3):
    """
    Apply edge detection to an input image using Canny or Sobel.

    Args:
    image (numpy.ndarray): Input image (color or grayscale)
    method (str): Edge detection method, either "canny" or "sobel" (default: "canny")
    canny_low (int): Lower threshold for Canny edge detection (default: 100)
    canny_high (int): Upper threshold for Canny edge detection (default: 200)
    sobel_ksize (int): Kernel size for Sobel operator (default: 3)

    Returns:
    numpy.ndarray: Image with detected edges
    """
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if method == "canny":
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
    elif method == "sobel":
        # Apply Sobel edge detection
        grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=sobel_ksize)
        grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=sobel_ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    else:
        raise ValueError("Method should be either 'canny' or 'sobel'.")

    return edges


def DINOv2_load():
    """Load the DINOv2 model and processor."""
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    return processor, model, device

def DINOv2_extract_image_features(image_path, processor, model, device, edge_method=None):
    """Extract features from an image, optionally applying edge detection."""
    image = Image.open(image_path).convert('RGB')

    # Convert the PIL image to a NumPy array for edge detection
    image_np = np.array(image)

    # Apply edge detection if specified
    if edge_method:
        image_np = apply_edge_detection(image_np, method=edge_method)

    # Convert the edge-detected NumPy array back to PIL image for the processor
    image = Image.fromarray(image_np)

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)
    return image_features


def display_features(features1, features2, title1, title2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(features1.cpu().numpy().flatten(), label=title1)
    axes[0].set_title(f'Features of {title1}')
    axes[0].set_xlabel('Feature Index')
    axes[0].set_ylabel('Feature Value')

    axes[1].plot(features2.cpu().numpy().flatten(), label=title2)
    axes[1].set_title(f'Features of {title2}')
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('Feature Value')

    plt.tight_layout()
    plt.show()


def calculate_similarity(features1, features2):
    """Calculate the cosine similarity between two feature vectors."""
    cos = nn.CosineSimilarity(dim=0)
    similarity = cos(features1[0], features2[0]).item()
    # Normalize similarity to range [0, 1]
    similarity = (similarity + 1) / 2
    return similarity

def display_similarity(image_path1, image_path2, similarity):
    """Display two images side by side with the similarity score."""
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Display first image
    ax[0].imshow(image1)
    ax[0].axis('off')
    ax[0].set_title('Image 1')

    # Display second image
    ax[1].imshow(image2)
    ax[1].axis('off')
    ax[1].set_title('Image 2')

    # Display the similarity score
    plt.suptitle(f'Similarity: {similarity * 100:.3f}%', fontsize=16)
    plt.show()

def get_embedding_function():
    """
    Purpose/Usage:
    This function creates and returns an embedding function for processing text using a model.

    Inputs:
    - None

    Outputs/Returns:
    - An object that can be used to compute embeddings for text using the OpenAIEmbeddings class.
    """
    embeddings = OpenAIEmbeddings()
    return embeddings

def extract_last_percentage(text):
    percentages = re.findall(r'\d+\.?\d*%', text)

    return percentages[-1] if percentages else None

def query_and_validate(EVAL_PROMPT: str, actual_answer: str, key_answer: str):
    """
    Purpose/Usage:
    This function queries a language model with a question and validates the response against an expected answer.

    Inputs:
    - actual_answer (str): The participant's answer.
    - key_answer (str): The correct answer.

    Outputs/Returns:
    - None
    """
    prompt = EVAL_PROMPT.format(
        kunci_jawaban=key_answer, jawaban_peserta=actual_answer
    )

    model = ChatOpenAI(model="gpt-4o-mini")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.content

    last_percentage = extract_last_percentage(evaluation_results_str_cleaned)

    print(evaluation_results_str_cleaned)
    print("============================")
    print(last_percentage)

    if last_percentage is None:
        return None  

    try:
        percentage_float = float(last_percentage.replace('%', ''))
        return percentage_float
    except ValueError:
        return None  

def grade_answer(answer_key, user_answer, tolerance=0.001):
    if answer_key is None or user_answer is None:
        return False
    return abs(answer_key - user_answer) <= tolerance

@app.route("/api/submit-answers", methods=["POST"])
@cross_origin()
def submit_answers():
    print("Form data received:", request.form)
    print("Files received:", request.files)

    def parse_float(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    answer1 = parse_float(request.form.get("answer1", ''))
    answer2 = parse_float(request.form.get("answer2", ''))
    answer3 = parse_float(request.form.get("answer3", ''))
    answer4 = request.form.get("answer4", '')
    answer5 = parse_float(request.form.get("answer5", ''))
    answer7 = parse_float(request.form.get("answer7", ''))
    answer8 = request.form.get("answer8", '')
    score = 0
    if 'scatter' in request.files:
        file = request.files['scatter']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            scatter = request.files['scatter']
    
            # Simpan atau proses file yang diunggah
            file_path1 = f"./uploads/{scatter.filename}"
            # scatter.save(file_path1)
            print("hasil-hasil")
            print(file_path1)
            processor, model, device = DINOv2_load()

            image_features1 = DINOv2_extract_image_features('./scatterplot(1).png', processor, model, device, edge_method="canny")
            image_features2 = DINOv2_extract_image_features(file_path1, processor, model, device, edge_method="canny")

            # Calculate similarity between all pairs of images
            sim12 = calculate_similarity(image_features1, image_features2)
            print('Similarity between image 1 and 2:', sim12)

            image_upload_message = f"Nilai simillarity gambar {sim12}"

            nilai_6 = float(sim12)

            if nilai_6 >= 0.83:
                score = score + 25
            else:
                score = score + 0

            # image_upload_message = f"3. File berhasil diunggah sebagai {filename}"
        else:
            image_upload_message = "3. Jenis file tidak valid"
    else:
        image_upload_message = "3. Tidak ada bagian file"


    # Debug output
    print("Answer1:", answer1)
    print("Answer2:", answer2)
    print("Answer3:", answer3)
    print("Answer4:", answer4)
    print("Answer5:", answer5)
    print("Answer7:", answer7)
    print("Answer8:", answer8)

    results = []
    

    if not grade_answer(slope, answer1):
        message1 = "1a. Salah"
    else:
        message1 = "1a. Benar"
        score = score + 25/6
    results.append(message1)

    if not grade_answer(intercept, answer2):
        message2 = "1a. Salah"
    else:
        message2 = "1a. Benar"
        score = score + 25/6
    results.append(message2)

    if not grade_answer(r_squared, answer3):
        message3 = "1b. Salah"
    else:
        message3 = "1b. Benar"
        score = score + 25/3

    results.append(message3)

    if not grade_answer(coefficient, answer5):
        message5 = "2. Salah"
    else:
        message5 = "2. Benar"
        score = score + 25
    results.append(message5)

    if not grade_answer(prediction, answer7):
        message7 = "4a. Salah"
    else:
        message7 = "4a. Benar"
        score = score + 25/2
    results.append(message7)

    EVAL_PROMPT = """
    Kunci Jawaban: {kunci_jawaban}
    Jawaban Peserta: {jawaban_peserta}
    ---
    Tinjau kesesuaian antara kunci jawaban dengan jawaban perserta per nomor.
    Berapa persentase jawaban peserta yang sesuai dengan kunci jawaban?
    """

    kunci_jawaban = """
    1. Persamaan regresi yang menyatakan hubungan antara biaya iklan (juta Rp) (x) dengan penjualan (juta Rp) (y) adalah y=5.2x-6
    2. Dari persamaan regresi di atas terlihat bahwa setiap kenaikan biaya iklan sebesar satu juta rupiah akan meningkatkan nilai penjualan sebesar 5.2 juta rupiah. Jika tidak ada biaya iklan maka nilai penjualannya menjadi negatif yaitu -6 juta rupiah. Tetapi hal ini tidak mungkin terjadi dalam kasus riil. 
    3. Model regresi mempunyai koefisien determinasi sebesar 0.9941 yang menunjukkan 99.41% variasi dalam penjualan dapat dijelaskan oleh hubungan linier dengan biaya iklan. Hal ini menunjukkan bahwa model regresi mempunya performa yang baik dalam memprediksi nilai penjualan berdasarkan besarnya biaya iklan."""
    
    nilai4 = query_and_validate(EVAL_PROMPT, answer4, kunci_jawaban)
    message4 = f"1c. {nilai4}"

    score = score + (25*nilai4)/300

    EVAL_PROMPT4 = """
    Kunci Jawaban: {kunci_jawaban}
    Jawaban Peserta: {jawaban_peserta}
    ---
    Tinjau kesesuaian antara kunci jawaban dengan jawaban perserta per nomor.
    Berapa persentase jawaban peserta yang sesuai dengan kunci jawaban?
    """

    kunci_jawaban4 = """
    Untuk meningkatkan hasil penjualan maka perusahaan harus meningkatkan biaya iklan dan biaya iklan harus lebih besar dari 1.1538 juta agar memperoleh nilai penjualan yang positif.."""
    
    nilai8 = query_and_validate(EVAL_PROMPT4, answer8, kunci_jawaban4)
    message8 = f"4b. {nilai8}"

    score = score + (25*nilai8)/200
    print(score)

    message = f"{message1}<br>{message2}<br>{message3}<br>{message4}<br>{message5}<br>{image_upload_message}<br>{message7}<br>{message8}"
    
    if len(results) == 5 and all(result != "Salah" for result in results):
        status = "success"
    else:
        status = "error"

    return jsonify({
        "status": status,
        "message": message,
        "score": score
    })

if __name__ == "__main__":
    app.run(debug=True)
