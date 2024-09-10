FROM python:3.11.9-slim 

WORKDIR /app-model

COPY ./requirements.txt /app-model/requirements.txt

RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
COPY . /app-model/
RUN chmod +x /app-model/run.sh

EXPOSE 5000 50051

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

CMD ["sh", "/app-model/run.sh"]
