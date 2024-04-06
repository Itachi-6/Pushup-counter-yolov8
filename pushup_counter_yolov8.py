from ultralytics import YOLO
import cv2
import torch
import math
import numpy as np
from PIL import Image

#Initializing the YOLO model
model = YOLO('yolov8l-pose.pt')

cap = cv2.VideoCapture('Video_path')

#Fucntion to calculate Angle between the points
def calculateAngle(p1, p2, p3):
   angle_p2p1 = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
   angle_p2p3 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])

   angle_p1p2p3 = abs(angle_p2p1 - angle_p2p3)
   ANGLE = math.degrees(angle_p1p2p3)

   if ANGLE > 180:
      ANGLE = 360 -ANGLE

   return ANGLE

counter = 0
position = None
direction = 0
posture = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break
    
    IMAGE = Image.fromarray(frame)

    img_width, img_height = IMAGE.size

    frame_result = model(source=frame, conf=0.5)

    #Extracting the keypoints which are essential to calculate the the pushup
    left_shoulder = (int(frame_result[0].keypoints[0].xy[0][5][0]), int(frame_result[0].keypoints[0].xy[0][5][1]))
    left_elbow = (int(frame_result[0].keypoints[0].xy[0][7][0]), int(frame_result[0].keypoints[0].xy[0][7][1]))
    left_wrist = (int(frame_result[0].keypoints[0].xy[0][9][0]), int(frame_result[0].keypoints[0].xy[0][9][1]))
    left_hip = (int(frame_result[0].keypoints[0].xy[0][11][0]), int(frame_result[0].keypoints[0].xy[0][11][1]))
    left_ankle = (int(frame_result[0].keypoints[0].xy[0][15][0]), int(frame_result[0].keypoints[0].xy[0][15][1]))

#----------------------------------------------------------------------------------------------------------------------
    # Drawing circles at the keypoints and joining them to visualize
    cv2.circle(frame, left_shoulder, 15, (255, 255, 255), 5)
    cv2.circle(frame, left_shoulder, 10, (0, 102, 255), 5)

    cv2.circle(frame, left_elbow, 15, (255, 255, 255), 5)
    cv2.circle(frame, left_elbow, 10, (0, 102, 255), 5)

    cv2.circle(frame, left_wrist, 15, (255, 255, 255), 5)
    cv2.circle(frame, left_wrist, 10, (0, 102, 255), 5)

    cv2.circle(frame, left_hip, 15, (255, 255, 255), 5)
    cv2.circle(frame, left_hip, 10, (0, 102, 255), 5)

    cv2.circle(frame, left_ankle, 15, (255, 255, 255), 5)
    cv2.circle(frame, left_ankle, 10, (0, 102, 255), 5)

    cv2.line(frame, left_wrist, left_elbow, (0, 0, 0), 8)
    cv2.line(frame, left_elbow, left_shoulder, (0, 0, 0), 8)
    cv2.line(frame, left_shoulder, left_hip, (0, 0, 0), 8)
    cv2.line(frame, left_hip, left_ankle, (0, 0, 0), 8)
#----------------------------------------------------------------------------------------------------------------------

    #Caculating the angle
    elbow_angle = calculateAngle(left_shoulder, left_elbow, left_wrist) #This is for write the logic to count the pushup
    hip_angle = calculateAngle(left_shoulder, left_hip, left_ankle) # This is for giving initial check to let the user know to correct the poster

    percentage = np.interp(elbow_angle, (85, 160), (100, 0))

    bar = np.interp(elbow_angle, (85, 160), (200, 600)) # This gives us the bar which goes from 0 to 100 with respect to the angle

    #This the logic/check which would increase the count for pushups
    if elbow_angle >= 160 and hip_angle >= 160:
        posture = 1
    
    if posture == 1:
        if elbow_angle >= 85:
            if direction == 0:
                counter += 0.5
                direction = 1
        
        if elbow_angle <= 85:
            if direction == 1:
                counter += 0.5
                direction = 0
    
    # This ensures the user is in correct posture before counting the pushups. If posture is not correct, the count will not be considered
    else:
        position = 'Correct the Posture'
        cv2.putText(frame, position, (500, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 128, 128), 6)
        
        
    cv2.rectangle(frame, (img_width-100, 600), (img_width-50, 200), (0, 0, 128), 2)
    if percentage == 100.0:
        cv2.rectangle(frame, (img_width-100, 600), (img_width-50, int(bar)), (128, 0, 128), -1) #changing the color to a different one, once the bar reaches 100
    else:
        cv2.rectangle(frame, (img_width-100, 600), (img_width-50, int(bar)), (0, 0, 128), -1)
    cv2.putText(frame, f'{int(percentage)}%', ((img_width-150, 380)), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 4)
    cv2.putText(frame, f'Counter - {str(int(counter))}', (25, 100), cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 128), 8)

    cv2.imshow('WINDOW', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()