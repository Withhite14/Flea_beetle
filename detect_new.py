import cv2
import time
import numpy as np
import sys

# Initialize camera index
camera_num = 0
cap = cv2.VideoCapture(camera_num)  # Start with default camera
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

prev_frame_time = 0
new_frame_time = 0
PROTOTXT = ('/home/white/Flea_beetle/Flea_beatle.cfg')  ################################# change path to config file #############################
MODEL = ('/home/white/Flea_beetle/Flea_beatle_last.weights')  ################################# change path to weights file #############################
confidence = 0.7
threshold = 0.4
labels = open('/home/white/Flea_beatle/Flea_beatle.names').read().strip().split('\n')
colors = np.random.uniform(0, 255, size=(len(labels), 3))
print('[Status] Loading Model...')
net = cv2.dnn.readNetFromDarknet(PROTOTXT, MODEL)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize variables for saving pictures and recording video
save_pictures = False  # Flag to check if we should save pictures
last_save_time = 0  # To track the last save time
save_interval = 10  # 10-second interval for saving pictures

record_video = False  # Flag to toggle video recording
video_writer = None  # Video writer object
recording_filename = None

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            if conf > confidence:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def make_prediction(net, layer_names, labels, image, confidences, threshold):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)
    return boxes, confidences, classIDs, idxs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(labels[classIDs[i]])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def switch_camera(camera_num):
    global cap
    cap.release()  # Release the current camera
    cap = cv2.VideoCapture(camera_num)  # Open the new camera
    if not cap.isOpened():
        print(f"[INFO] Camera {camera_num} not available, switching back to camera 0.")
        camera_num = 0  # Reset to camera 0 if the selected camera is unavailable
        cap = cv2.VideoCapture(camera_num)
    else:
        print(f"[INFO] Switched to camera {camera_num}")
    return camera_num

while True:
    rec, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence, threshold)
    image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    image = cv2.resize(image, (1080, 720), interpolation=cv2.INTER_AREA)
    cv2.putText(img=image, text=str(len(idxs)), org=(36, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=2)

    # Handle picture saving
    if save_pictures:
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_filename = f"capture_{int(current_time)}.jpg"
            cv2.imwrite(save_filename, image)
            print(f"[INFO] Image saved as {save_filename}")
            last_save_time = current_time  # Reset the last save time

    # Handle video recording
    if record_video and video_writer is not None:
        video_writer.write(image)  # Write the current frame to the video file

    # Show the video feed
    cv2.imshow("Detector", image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == ord('p'):  # 'P' key to toggle picture saving
        save_pictures = not save_pictures
        print(f"[INFO] Picture saving {'enabled' if save_pictures else 'disabled'}")
    elif key == ord('o'):  # 'O' key to toggle video recording
        if record_video:
            record_video = False
            video_writer.release()
            print(f"[INFO] Video recording stopped and saved as {recording_filename}")
        else:
            recording_filename = f"record_{int(time.time())}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(recording_filename, fourcc, 20.0, (1080, 720))
            record_video = True
            print(f"[INFO] Video recording started as {recording_filename}")
    elif key == ord('c'):  # 'C' key to increment camera index
        camera_num += 1
        camera_num = switch_camera(camera_num)
    elif key == ord('x'):  # 'X' key to decrement camera index
        camera_num -= 1
        if camera_num < 0:
            camera_num = 0  # Prevent negative index
        camera_num = switch_camera(camera_num)

# Cleanup
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
