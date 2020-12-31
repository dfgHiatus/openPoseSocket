# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
from imutils.video import VideoStream
from statistics import mean
import requests
import asyncio
import websockets
import socket
import numpy as np
import urllib.request 
import datetime
import imutils
import math
import dlib
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=225, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=175, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height



# Load video input. Be sure the web camera you're using is the default device!
# my_ip = '192.168.1.251:8080'
my_ip = input("Please enter your IP Webcamera IP and Port (EX: 192.168.1.251:8080): ")
# cap = cv2.VideoCapture('http://192.168.1.251:8080/')

# Initiallize eye tracking for screen mode
# gaze = GazeTracking()

print("Loading AI Stuff...")
# AI Stuff, loading the face database
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
cap = cv2.VideoCapture(args.input if args.input else 0)
print("Done Loading.")

# Websocket runs on port 7001 by default
print("Creating Websocket Port....")
print("Websocket created on port 7001.\n")
print("Initiallizing body tracking. Landmarks are displayed below upon connection. \n")
socketString = ""

async def facetrack(websocket, path):
    async for message in websocket:
    
        socketString = ""
        # hasFrame, frame = cap.read()
        frame = imutils.url_to_image('http://'+my_ip+'/shot.jpg') # 'Load it as it is'

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
    
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5,  127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            if conf > args.thr:
                socketString += "[" + str(int(x)) + ";" + str(int(y)) + "],"
                points.append((int(x), int(y)))
            else:
                socketString += "[0;0},"
                points.append(None)
            
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.    FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('OpenPose using OpenCV', frame)
    
        print(socketString)
        await websocket.send(socketString)

# Pushes string to port 7000
asyncio.get_event_loop().run_until_complete(
    websockets.serve(facetrack, 'localhost', 7001))
asyncio.get_event_loop().run_forever()    