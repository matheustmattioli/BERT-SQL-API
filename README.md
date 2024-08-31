# BERT model for SQL Injection

Doing a simple api with flask and BERT model for the SQL Injection Dataset https://www.kaggle.com/datasets/sajid576/sql-injection-dataset.

Model code is based on https://www.kaggle.com/code/bobaaayoung/bert-word2vec-lstm-cnn-text-classification-ipynb.

## How to use 

Download the model weights and save it in ```bert-sql-api/artifact/```. The steps are

```
# clone this repo
cd bert-sql-api
mkdir artifact
wget -P artifact 'https://drive.google.com/file/d/1m8HPQLToDfvE9tIE_cisuG8qp9b-lntf'
```

[***Trained model weights available here.***](https://drive.google.com/file/d/1m8HPQLToDfvE9tIE_cisuG8qp9b-lntf).

### Local

Install all the requirements from requirements.txt in local machine

```
pip install --no-cache-dir -r requirements.txt
```

and run the application ```python app.py```

### Docker
Alternatively run the dockerfile

```
docker run -p 5000:5000 gkdogifjhif/bert-sql-api:latest
```