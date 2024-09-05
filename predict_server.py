# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grpc, server.predict_pb2, server.predict_pb2_grpc
from concurrent import futures
from model.prediction import run_prediction

class PredictionService(server.predict_pb2_grpc.PredictionServiceServicer):
    def Predict(self, request, context):
        # receive the request input (json) and run the prediction
        input = request.input
        result = run_prediction(input)
        return server.predict_pb2.PredictResponse(result=result)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.predict_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC Server started. Listening on port 50051...")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()