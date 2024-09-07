from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import re

os.environ['OPENAI_API_KEY'] = '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'
app = Flask(__name__)

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

    model = ChatOpenAI(model="gpt-4")
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

def grade_answer(answer_key, user_answer, tolerance=0.01):
    if answer_key is None or user_answer is None:
        return False
    return abs(answer_key - user_answer) <= tolerance
    
@app.route("/api/submit-answers", methods=["POST"])
def submit_answers():
    data = request.json
    answer1 = data.get("answer1")
    answer2 = data.get("answer2")
    answer3 = data.get("answer3")
    answer4 = data.get("answer4")
    answer5 = data.get("answer5")
    answer7 = data.get("answer7")

    print(answer1)
    print(answer2)
    print(answer3)
    print(answer4)
    print(answer5)
    print(answer7)

    incorrect_messages = []
    total_score = 0

    if not grade_answer(slope, answer1):
        incorrect_messages.append("Slope salah")
    else:
        total_score += 25 / 3

    if not grade_answer(intercept, answer2):
        incorrect_messages.append("Intercept salah")
    else:
        total_score += 25 / 3

    if not grade_answer(r_squared, answer3):
        incorrect_messages.append("R-Squared salah")
    else:
        total_score += 25 / 3

    if not grade_answer(coefficient, answer5):
        incorrect_messages.append("Coefficient Correlation salah")
    else:
        total_score += 25

    if not grade_answer(prediction, answer7):
        incorrect_messages.append("Prediction salah")
    else:
        total_score += 25 / 2

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
    

    nilai = query_and_validate(EVAL_PROMPT, answer4, kunci_jawaban)

    if not incorrect_messages:
        status = "success"
        message = f"Semua jawaban benar! Total: {total_score:.2f}."
    else:
        status = "error"
        message = f"{len(incorrect_messages)} jawaban salah: " + ", ".join(incorrect_messages) + f". Total: {total_score:.2f}. Similarity string: {nilai}"

    return jsonify({
        "status": status,
        "message": message,
    })

if __name__ == "__main__":
    app.run(debug=True)