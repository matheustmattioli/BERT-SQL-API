import torch, json, re, urllib.parse, base64
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

batch_size = 64
max_length = 128
hidden_size = 128
num_layers = 1
num_classes = 2
num_epochs = 5
output_size = num_classes  # Number of classes in your classification task

class PreProcess:
    def __init__(self):
        pass

    def decode_sql(self, encoded_string):
        decoded_string = encoded_string
        try:
            decoded_string = bytes.fromhex(encoded_string).decode('ascii')
        except:
            pass
        try:
            decoded_string = bytes.fromhex(encoded_string).decode('unicode_escape')
        except:
            pass
        try:
            decoded_string = json.loads(encoded_string)
        except:
            pass
        try:
            decoded_string = urllib.parse.unquote(encoded_string)
        except:
            pass
        try:
            decoded_string = base64.b64decode(encoded_string).decode('utf-8')
        except:
            pass
        return decoded_string

    def lowercase_sql(self, query):
        return query.lower()

    def generalize_sql(self, query):
        generalized_query = re.sub(r'\d+', '0', query)
        return generalized_query

    def tokenize_sql(self, query):
        query = re.sub(r'([<>!=])', r' \1 ', query)
        tokens = query.split()
        return ' '.join(tokens)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        } 

# BERT+TextCNN
class BertTextCNNClassifier(nn.Module):
    def __init__(self, bert_model, num_filters, filter_sizes, output_size):
        super(BertTextCNNClassifier, self).__init__()
        self.bert_model = bert_model
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=bert_model.config.hidden_size, out_channels=num_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_size)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Transpose to fit the Conv1d input format (batch_size, num_channels, seq_len)
        embedded = outputs.last_hidden_state.transpose(1, 2)

        # Apply convolutional layers with different kernel sizes and max-pooling
        pooled_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = nn.functional.relu(conv_layer(embedded))
            pooled_out, _ = torch.max(conv_out, dim=2)
            pooled_outputs.append(pooled_out)

        # Concatenate pooled outputs and flatten
        pooled_outputs = torch.cat(pooled_outputs, dim=1)
        pooled_outputs = self.dropout(pooled_outputs)

        # Fully connected layer for classification
        logits = self.fc(pooled_outputs)
        return logits
    

if __name__ == '__main__':
    # Opendataset
    df = pd.read_csv("data/Modified_SQL_Dataset.csv").sample(10000).reset_index(drop=True)
    
    # Preprocess
    preprocess = PreProcess()
    df['Text'] = df['Query'].apply(preprocess.decode_sql)  
    df['Text'] = df['Text'].apply(preprocess.lowercase_sql)  
    df['Text'] = df['Text'].apply(preprocess.generalize_sql)  
    df['Text'] = df['Text'].apply(preprocess.tokenize_sql)  

    # Split the dataset
    train_df,test_df = train_test_split(df,test_size=0.20,random_state=50,shuffle=True)

    train_texts, train_labels = train_df['Text'].tolist(), train_df['Label'].tolist()
    test_texts, test_labels = test_df['Text'].tolist(), test_df['Label'].tolist()

    # BERT tokenizer
    bert_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model = BertModel.from_pretrained(bert_model_name)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Bert TextCNN
    num_filters = 100
    filter_sizes = [2, 3, 4] 
    model = BertTextCNNClassifier(bert_model, num_filters, filter_sizes, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Train the model
    print("Start training")
    for epoch in range(num_epochs):
        total_loss = 0
        correct_train = 0
        total_train = 0

        with tqdm(train_loader, unit="batch") as t:
            for batch in t:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                t.set_postfix({'loss': total_loss / (t.n + 1), 'accuracy': correct_train / total_train})

    # Save the model
    torch.save(model.state_dict(), 'artifact/bert_textcnn_classifier.pth')

    # Test the model
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []

        with tqdm(test_loader, unit="batch") as t:
            for batch in t:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                _, predicted = torch.max(logits.data, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                t.set_postfix({})

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
