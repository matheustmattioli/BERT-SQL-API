from flask import Flask, request, jsonify
import os, grpc, server.predict_pb2, server.predict_pb2_grpc

# Setting initial configurations
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"message": "Bad Request: No JSON data received"}), 400
    input_text = data.get('input')
    if not input_text:
        return jsonify({"message": "Bad Request: 'input' field is required"}), 400
    try:
        # make gRPC call
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = server.predict_pb2_grpc.PredictionServiceStub(channel)
            response = stub.Predict(server.predict_pb2.PredictRequest(input=input_text))
        # process gRPC response and return to the Flask app
        prediction = {
            'is_attack': True if response.result == 'SQL Injection' else False,
            'type_attack':  response.result 
        }
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'message': f"Error: {e}"})

if __name__ == '__main__':
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host=host, port=port)

