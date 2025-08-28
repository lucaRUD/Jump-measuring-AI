import cv2
import mediapipe as mp
from collections import deque
import numpy as np
import math

def angle_from_three_points(a, b, c):
    #calculate the angle between 3 points, angle is returned at b in degrees
    v1 = (a[0] - b[0] ,a[1] - b[1]) #vector from b to a (knee to hip)
    v2 = (c[0] - b[0], c[1] - b[1]) #vector from b to c (ankle to hip)
    dot_product = v1[0]*v2[0]+ v1[1] *v2[1] #dot product (produs scalar)
    mag1 = math.hypot(v1[0], v1[1]) #magnitude of vector 1 --> modulul lui v1 | sqrt(x^2+y^2)
    mag2 = math.hypot(v2[0], v2[1]) #magnitude of vector 2 --> modulul lui v2

    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_theta = dot_product / (mag1 * mag2)

    #numeric safety
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_rad = math.acos(cos_theta) #angle in radians
    return math.degrees(theta_rad)  #convert to degrees

    


# 0 = camera default (laptop)
cap = cv2.VideoCapture("manjump3-47.1.mp4")


WARMUP_FRAMES = 20
VELOCITY_THRESHOLD = 1.5  # pixels per frame
ANGLE_THRESHOLD = 160  # degrees, knee angle to consider standing
MIN_BASELINE_SAMPLES = 5
baseline_samples = []  # a list where we'll store candidate "hip_y" values
frame_index = 0 # counts which frame we are on
prev_hip_y = None  # remembers hip_y from the previous frame
baseline_y = None   # the final baseline value (starts as unknown)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Smoothing buffer for hip y
hip_y_buffer = deque(maxlen=5) # simple moving average over last 5 frames
baseline_y = None               
last_y = None                   # for velocity later

measuring = False
peak_y = None
max_height_px = 0

if not cap.isOpened():
    print("Video could not be opened")
    exit()


while True:
    ret, frame = cap.read()  # Read a frame    
    if not ret:
        break


    # Convert the frame to RGB because Mediapipe requires RGB images and cv2 outputs BGR
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #process frame with Pose model
    results = pose.process(frame_rgb)

    #draw landmarks
    if results.pose_landmarks:
        h , w = frame.shape[:2]
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        LHIP = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        RHIP = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        
        LSHOULDER= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        LKNEE= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        LANKLE= results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        l_shoulder = int(LSHOULDER.x * w), int(LSHOULDER.y * h)#left shoulder in pixels
        l_hip = int(LHIP.x * w), int(LHIP.y * h)#left hip in pixels
        l_knee = int(LKNEE.x * w), int(LKNEE.y * h)
        l_ankle = int(LANKLE.x * w), int(LANKLE.y * h)

        knee_angle = angle_from_three_points(l_shoulder, l_knee, l_ankle)
        cv2.putText(frame , f"Knee angle: {knee_angle:.1f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        #mid hip x in pixels 
        hip_x_px = int(((LHIP.x + RHIP.x) / 2) * w)

        #mid hip y in pixels 
        hip_y_px = int(((LHIP.y + RHIP.y) / 2) * h)

        hip_y_buffer.append(hip_y_px)

        #smooth hip_y
        hip_y = sum(hip_y_buffer) / len(hip_y_buffer)


                
        cv2.circle(frame, (hip_x_px, hip_y_px), 6, (255 , 0, 0), -1)

        #automatic baseline setter 
        frame_index += 1 #count the frames

        if prev_hip_y is None:
            vel = 0
        else:
            vel = abs(hip_y - prev_hip_y)

        prev_hip_y = hip_y # update memory for next loop


        if baseline_y is None:  #only works till baseline is found 
            if frame_index <= WARMUP_FRAMES: #collect only warmup frames
                if vel < VELOCITY_THRESHOLD and knee_angle > ANGLE_THRESHOLD: #if the hip is stable and knee angle is bigger than knee_angle degrees 
                    baseline_samples.append(hip_y)
            else:
                # decide baseline after warmup
                if len(baseline_samples) > MIN_BASELINE_SAMPLES: 
                    baseline_y = float(np.mean(baseline_samples))
                    print("Baseline estimated to be at " "%.2f" % baseline_y, "px")
                    measuring = True
                else:
                    # fallback if we didn't find enough samples where the hip is not moving 
                    baseline_y = float(hip_y)
                    print("Baseline fallback to current hip position:", "%.2f" % baseline_y, "px")

        if measuring :
            if peak_y is None or hip_y < peak_y : #smaller y means its higher, it measures y from the top
                peak_y = hip_y # new peak
                max_height_px = baseline_y - peak_y
                cv2.putText(frame , f"Jump height: {max_height_px}px", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # if baseline is set, draw it as a horizontal line

        if baseline_y is not None:
            cv2.line(frame, (0, int(baseline_y)), (w, int(baseline_y)), (0, 255, 0), 2)




        




    


    cv2.imshow('Pose Detection', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        baseline_y = hip_y #set standing baseline
        print("Baseline set to:", "%.2f" % baseline_y)
        measuring = True
        peak_y = hip_y
        max_height_px = 0
        print("Jump measurement started")
    elif key == ord('r'):
        measuring = False
        peak_y = None
        max_height_px = 0
        print("Jump measurement reset")

    elif key == ord('q'):
        break
print("Jump height is:", "%.2f" % max_height_px, "px")
cap.release()
cv2.destroyAllWindows()



















# This code allows you to click on an image to save the coordinates of the points clicked.
# image = cv2.imread('man-jumping.png')

# coordinates = []

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         coordinates.append((x, y))
#         img_copy = image.copy()

#         for i, coord in enumerate(coordinates):
#             cv2.circle(img_copy, coord, 20, (0, 255, 0), -1)
#             cv2.putText(img_copy, f'{i + 1}', (coord[0] , coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         for i, point in enumerate(coordinates):

#             if i > 0:
#                 cv2.line(img_copy, coordinates[i - 1], point, (255, 0, 0), 2)   

#         cv2.imshow('Imaginea mea', img_copy)

#         print(f'Saved coordinates: {x}, {y}')  # Corrected print statement

#         #save in file 
#         with open('coordinates.txt', 'a') as f:
#             f.write(f'{x}, {y}\n')



# cv2.imshow('Imaginea mea', image)
# cv2.setMouseCallback('Imaginea mea', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
