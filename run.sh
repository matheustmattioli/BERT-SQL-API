#!/bin/bash

# Add your script code here
python predict_server.py & > /dev/null
sleep 5
python app.py & > /dev/null

wait