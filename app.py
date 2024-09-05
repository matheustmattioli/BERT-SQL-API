from flask import Flask, request, jsonify
import os, grpc, server.predict_pb2, server.predict_pb2_grpc

# Setting initial configurations
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # make gRPC call
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = server.predict_pb2_grpc.PredictionServiceStub(channel)
            response = stub.Predict(server.predict_pb2.PredictRequest(input=request.form['input']))
        # process gRPC response and return to the Flask app
        prediction = {'result': response.result}
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'message': f"Error: {e}"})

if __name__ == '__main__':
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    app.run(host=host, port=port)
    app.run(debug=True)

