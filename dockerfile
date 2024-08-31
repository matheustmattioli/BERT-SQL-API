FROM python:3.11.9-slim 

WORKDIR /app-model

COPY . /app-model/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

CMD ["python", "/app-model/app.py"]

