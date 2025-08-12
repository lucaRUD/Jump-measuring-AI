import cv2

# 0 = camera default (laptop)
cap = cv2.VideoCapture("manjump.mp4")


if not cap.isOpened():
    print("Video could not be opened")
    exit()


while True:
    ret, frame = cap.read()  # Read a frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
    
    if not ret:
        break

    cv2.imshow('Camera mea', gray)

    # Dacă apeși tasta 'q', ieși din loop
    if cv2.waitKey(75) & 0xFF == ord('q'):
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