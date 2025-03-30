import os
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Define the correct file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
FILE_PATH = os.path.join(BASE_DIR, 'disease.xlsx')  # Ensure full path is used

# Check if the Excel file exists before loading
if os.path.exists(FILE_PATH):
    try:
        df = pd.read_excel(FILE_PATH)
        print("✅ Disease database loaded successfully!")
    except Exception as e:
        print(f"❌ Error reading 'disease.xlsx': {e}")
        df = pd.DataFrame(columns=['Disease', 'Information', 'Symptoms', 'Treatment', 'Precaution'])
else:
    print("⚠️ Warning: 'disease.xlsx' file not found. API will return 'Database not available'.")
    df = pd.DataFrame(columns=['Disease', 'Information', 'Symptoms', 'Treatment', 'Precaution'])  # Empty fallback

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/get_disease', methods=['POST'])
def get_disease():
    name = request.json.get('name')
    disease_name = request.json.get('disease')

    if df.empty:
        return jsonify({'error': '⚠️ Disease database is unavailable. Please upload disease.xlsx'}), 500

    result = df.loc[df['Disease'].str.lower() == disease_name.lower()]

    if not result.empty:
        data = {
            'Name': name,
            'Disease': result.iloc[0]['Disease'],
            'Information': result.iloc[0]['Information'],
            'Symptoms': result.iloc[0]['Symptoms'],
            'Treatment': result.iloc[0]['Treatment'],
            'Precaution': result.iloc[0]['Precaution']
        }
        return jsonify(data)
    else:
        return jsonify({'error': 'Disease not found'}), 404

@app.route('/generate_report', methods=['POST'])
def generate_report():
    name = request.json.get('name', 'Unknown')
    disease_name = request.json.get('disease')

    if df.empty:
        return jsonify({'error': '⚠️ Disease database is unavailable. Please upload disease.xlsx'}), 500

    result = df.loc[df['Disease'].str.lower() == disease_name.lower()]

    if not result.empty:
        disease = result.iloc[0]['Disease']
        info = result.iloc[0]['Information']
        symptoms = result.iloc[0]['Symptoms']
        treatment = result.iloc[0]['Treatment']
        precaution = result.iloc[0]['Precaution']

        # Secure filename
        safe_name = ''.join(c for c in name if c.isalnum())
        safe_disease = ''.join(c for c in disease if c.isalnum())
        file_path = f"{safe_name}_{safe_disease}_report.pdf"

        c = canvas.Canvas(file_path, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 20)
        c.drawString(30, height - 50, f"Medical Report for {name}")

        c.setFont("Helvetica", 16)
        c.drawString(30, height - 90, f"Disease: {disease}")

        c.setFont("Helvetica", 14)
        c.drawString(30, height - 130, f"Information: {info}")
        c.drawString(30, height - 180, f"Symptoms: {symptoms}")
        c.drawString(30, height - 230, f"Treatment: {treatment}")
        c.drawString(30, height - 280, f"Precaution: {precaution}")

        c.save()

        return send_file(file_path, as_attachment=True)

    else:
        return jsonify({'error': 'Disease not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
