import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prev_time = 0
curr_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandmark.landmark):
                print(id, landmark)
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                print(f"Id: {id}\nX: {cx}\nY: {cy}")
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
