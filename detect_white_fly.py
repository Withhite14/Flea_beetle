import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# Initialize camera index
camera_num = 0
cap = cv2.VideoCapture(camera_num)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

prev_frame_time = 0
new_frame_time = 0

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use appropriate path to your YOLOv8 model weights

# Initialize variables for saving pictures and recording video
save_pictures = False  # Flag to check if we should save pictures
last_save_time = 0  # To track the last save time
save_interval = 10  # 10-second interval for saving pictures

record_video = False  # Flag to toggle video recording
video_writer = None  # Video writer object
recording_filename = None

# Switch the camera by releasing the current camera and starting a new one
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
    if not rec:
        print("[ERROR] Failed to read from the camera.")
        break
    
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Perform YOLOv8 inference
    results = model(image)

    # Draw bounding boxes on the image
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Get box coordinates
        confs = result.boxes.conf.cpu().numpy()  # Confidence scores
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            conf = confs[i]
            class_id = cls_ids[i]
            
            # Draw the bounding box
            color = (0, 255, 0)  # Green box for detections
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{model.names[class_id]} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Handle picture saving
    if save_pictures:
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_filename = f"capture_{int(current_time)}.jpg"
            cv2.imwrite(save_filename, image)
            print(f"[INFO] Image saved as {save_filename}")
            last_save_time = current_time

    # Handle video recording
    if record_video and video_writer is not None:
        video_writer.write(image)  # Write the current frame to the video file

    # Show the video feed
    cv2.imshow("YOLOv8 Detector", image)

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
            video_writer = cv2.VideoWriter(recording_filename, fourcc, 20.0, (image.shape[1], image.shape[0]))
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
