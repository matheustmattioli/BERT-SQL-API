import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
model = MobileBertForSequenceClassification.from_pretrained('cssupport/mobilebert-sql-injection-detect').to(device)
model.eval()

def predict(text):
    inputs = tokenizer(text, padding=False, truncation=True, return_tensors='pt', max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class, probabilities[0][predicted_class].item()


# text = "SELECT * FROM users WHERE username = 'admin' AND password = 'password';"
# text = "select * from users where username = 'admin' and password = 'password';"
# text = "SELECT * from USERS where id  =  '1' or @ @1  =  1 union select 1,version  (    )   -- 1'"
# text = "select * from data where id  =  '1'  or @"
# text ="select * from users where id  =  1 or 1#\"?  =  1 or 1  =  1 -- 1"
# text = "curl -s -X POST http://localhost:5000/api/v1/login -d 'username=admin&password=password'"
text = "curl -s -X POST https://api.magalu.cloud/compute/v0/instances -H 'Content-Type:application/json' -d 'description': '1\/%\/27%20ORDER%20BY%203--%2B'"
predicted_class, confidence = predict(text)

if predicted_class > 0.7:
    print("Prediction: SQL Injection Detected")
else:
    print("Prediction: No SQL Injection Detected")
    
print(f"Confidence: {confidence:.2f}")
# OUTPUT
# Prediction: SQL Injection Detected
# Confidence: 1.00
