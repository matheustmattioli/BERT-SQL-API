from model.model import BertTextCNNClassifier, PreProcess
from transformers import BertTokenizer, BertModel
import torch

## set the device
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

## Hyperparameters
num_filters, filter_sizes, output_size = 100, [2, 3, 4], 2

## Load the machine learning model
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name) # define the tokenizer
bert_model = BertModel.from_pretrained(bert_model_name)
model = BertTextCNNClassifier(bert_model, num_filters, filter_sizes, output_size)
model.load_state_dict(torch.load('./artifact/bert_textcnn_classifier.pth', map_location=torch.device(device_type)))
model.to(device)
# Warming up the model
dummy_input = torch.zeros(1, 128).long().to(device)
dummy_mask = torch.ones(1, 128).long().to(device)
with torch.no_grad():
    _ = model(dummy_input, dummy_mask)
model.eval()

def run_prediction(input_list):
    # Receive the input and preprocess it
    preprocessor = PreProcess()
    
    input_preprocessed_list = []
    for input_text in input_list:
        input_preprocessed = preprocessor.decode_sql(input_text)    
        input_preprocessed = preprocessor.lowercase_sql(input_preprocessed)
        input_preprocessed = preprocessor.generalize_sql(input_preprocessed)
        input_preprocessed = preprocessor.tokenize_sql(input_preprocessed)
        input_preprocessed_list.append(input_preprocessed)

    # Tokenize the input
    encoding = tokenizer(
        input_preprocessed_list,
        truncation=True,
        padding='max_length',
        max_length=128,  # Using the same max_length as in training
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Classify the input
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()

    # Return the prediction
    predictions = ['SQL Injection' if p == 1 else 'No SQL Injection' for p in predictions]

    return predictions
