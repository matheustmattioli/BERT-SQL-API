from flask import Flask, render_template, request
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch
# import joblib

app = Flask(__name__)

# Load the machine learning model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
# tokenizer.decode(clean_up_tokenization_spaces=True)
model = MobileBertForSequenceClassification.from_pretrained('cssupport/mobilebert-sql-injection-detect').to(device)
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['input1']
    # text = "curl -s -X POST https://api.magalu.cloud/compute/v0/instances -H 'Content-Type:application/json' -d 'description': '1\/%\/27%20ORDER%20BY%203--%2B'"
    inputs = tokenizer(text, padding=False, truncation=True, return_tensors='pt', max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    prediction = "SQL Injection Detected" if predicted_class > 0.7 else "No SQL Injection Detected"
    confidence = probabilities[0][predicted_class].item()

    return render_template('result.html', prediction=prediction)
    # return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)