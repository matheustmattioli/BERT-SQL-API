from flask import Flask, request, jsonify
import os, grpc, server.predict_pb2, server.predict_pb2_grpc

# Setting initial configurations
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = [value for key, value in data.items()]
    if data is None:
        return jsonify({"message": "Bad Request: No JSON data received"}), 400
    if not inputs:
        return jsonify({"message": "Bad Request: 'input' field is required"}), 400
    try:
        # make gRPC call
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = server.predict_pb2_grpc.PredictionServiceStub(channel)
            response = stub.Predict(server.predict_pb2.PredictRequest(input=inputs))
        # process gRPC response and return to the Flask app
        predictions = {
            f'result_{i}': {
                'is_attack': True if result == 'SQL Injection' else False,
                'type_attack':  result if result == 'SQL Injection' else 'No Threat'
            }
            for i, result in enumerate(response.result)
        }
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'message': f"Error: {e}"})

if __name__ == '__main__':
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host=host, port=port)

