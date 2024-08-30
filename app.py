from flask import Flask, render_template, request
from model import BertTextCNNClassifier, PreProcess, num_filters, filter_sizes, output_size
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the machine learning model
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name) # define the tokenizer
bert_model = BertModel.from_pretrained(bert_model_name)
model = BertTextCNNClassifier(bert_model, num_filters, filter_sizes, output_size)
model.load_state_dict(torch.load('artifact/bert_textcnn_classifier.pth'))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive the input and preprocess it
    input_text = request.form['input1']
    preprocessor = PreProcess()

    input_preprocessed = preprocessor.decode_sql(input_text)    
    input_preprocessed = preprocessor.lowercase_sql(input_preprocessed)
    input_preprocessed = preprocessor.generalize_sql(input_preprocessed)
    input_preprocessed = preprocessor.tokenize_sql(input_preprocessed)

    # Tokenize the input
    # TODO: Tokenize the input using the tokenizer

    # Classify the input
    # TODO: Classify the input using the model

    # Return the prediction
    prediction = 1 # 1 is SQL Injection, 0 is No SQL Injection
    prediction = 'SQL Injection' if prediction == 1 else 'No SQL Injection'

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

# Tests
# text = "curl -s -X POST https://api.magalu.cloud/compute/v0/instances -H 'Content-Type:application/json' -d 'description': '1\/%\/27%20ORDER%20BY%203--%2B'"
