import cv2
import mediapipe as mp
from collections import deque
import numpy as np

# 0 = camera default (laptop)
cap = cv2.VideoCapture("manjump2.mp4")


WARMUP_FRAMES = 20
VELOCITY_THRESHOLD = 1.5  # pixels per frame
baseline_samples = []  # a list where we'll store candidate "hip_y" values
frame_index = 0 # counts which frame we are on
prev_hip_y = None  # remembers hip_y from the previous frame
baseline_y = None   # the final baseline value (starts as unknown)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Smoothing buffer for hip y
hip_y_buffer = deque(maxlen=5) # simple moving average over last 5 frames
baseline_y = None               # will set by pressing 'b'
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
                if vel < VELOCITY_THRESHOLD: #if the hip is stable 
                    baseline_samples.append(hip_y)
            else:
                # decide baseline after warmup
                if len(baseline_samples) > 5: 
                    baseline_y = float(np.mean(baseline_samples))
                    print("Baseline estimated to be at" "%.2f" % baseline_y, "px")
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
