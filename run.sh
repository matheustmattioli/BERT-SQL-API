#!/bin/bash

python predict_server.py & > /dev/null

GRPC_SERVER="localhost"
GRPC_PORT=50051

echo "gRPC server: $GRPC_SERVER:$GRPC_PORT is starting... Wait a few seconds"

while ! nc -z $GRPC_SERVER $GRPC_PORT; do   
    echo "Waiting for gRPC server to start..."
    sleep 10
done
echo "gRPC server is up!"

python app.py & > /dev/null

wait