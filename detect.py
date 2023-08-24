#!/usr/bin/env python3

import cv2
import time
import argparse
import numpy as np
import sys

# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--camera", required=True,
# 	help="select camera")

num = 0
cap = cv2.VideoCapture(int(num))
cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
prev_frame_time = 0
new_frame_time = 0
PROTOTXT = ('/home/white/Flea_beetle/Flea_beatle.cfg') ################################# change path to config file #############################
MODEL = ('/home/white/Flea_beetle/Flea_beatle_last.weights') ################################# change path to weights file #############################
confidence = 0.7
threshold = 0.4
#Initialize Objects and corresponding colors which the model can detect
labels = open('/home/white/Flea_beetle/Flea_beatle.names').read().strip().split('\n')
# labels = ["hand"] ############################## put labels name #####################################
colors = np.random.uniform(0, 255, size=(len(labels), 3))
#Loading Model
print('[Status] Loading Model...')
net = cv2.dnn.readNetFromDarknet(PROTOTXT, MODEL)
# Get the ouput layer names
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def make_prediction(net, layer_names, labels, image, confidences, threshold):
    height, width = image.shape[:2]
    
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            text = "{}".format(labels[classIDs[i]])

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # print(text)

    return image



while True: 
    rec, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence, threshold)
    image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fpsx = float(fps)
    # print(len(idxs))
    image = cv2.resize(image, (1080,720), interpolation=cv2.INTER_AREA)
    cv2.putText(img=image, text=str(len(idxs)), org=(36, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255),thickness=2)
    start_point = (0, 690)
    end_point = (170, 720)
    color = (0, 0, 0)
    thickness = -1
    # cv2.rectangle(image, start_point, end_point, color, thickness)
    # cv2.putText(img=image, text="camera:"+str(num), org=(5, 715), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 255),thickness=1)
    cv2.imshow("Detector",image)

    if cv2.waitKey(1) == 27: 
        break
        
