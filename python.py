import cv2
import mediapipe as mp
from collections import deque

# 0 = camera default (laptop)
cap = cv2.VideoCapture("manjump.mp4")


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Smoothing buffer for hip y
hip_y_buffer = deque(maxlen=5) # simple moving average over last 5 frames
baseline_y = None               # will set by pressing 'b'
last_y = None                   # for velocity later


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
        hip_y_smooth = sum(hip_y_buffer) / len(hip_y_buffer)

        cv2.circle(frame, (hip_x_px, hip_y_px), 6, (0 , 0, 255), -1)

        # if baseline is set, draw it as a horizontal line

        if baseline_y is not None:
            cv2.line(frame, (0, int(baseline_y)), (w, int(baseline_y)), (0, 255, 0), 2)




        




    


    cv2.imshow('Pose Detection', frame)

    key = cv2.waitKey(90) & 0xFF
    if key == ord('b'):
        baseline_y = hip_y_smooth #set standing baseline
        print("Baseline set to:", baseline_y)
    elif key == ord('q'):
        break

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