import cv2
import time
import numpy as np
import math
import HandTrackingModule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# Parameters____________________
capWidth, capHeight = 640, 480
# ______________________________

cap = cv2.VideoCapture(0)

# Changing Resolution
cap.set(3, capWidth)
cap.set(4, capHeight)

# Initializing Time
currTime = 0
prevTime = 0

# Creating Object for Hand Detection Module
detector = htm.handDetector(detectionCon=0.7)

# Initialization of pycaw for dealing with volume of device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

volBar = 400   # Base of Volume Bar in Display
volPer = 0     # Volume percentage

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findposition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])  # 4 --> Thumb-Tip, 8 --> IndexFinger-Tip

        # Getting Coordinates of the above landmarks
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Circling them for representation
        cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 255, 0), cv2.FILLED)

        # Getting centre of Line
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        # Creating a line between both the points
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Range of Hand  = 50 to 300
        # Range of Volume = -65 to 0
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(volPer)
        volume.SetMasterVolumeLevel(vol, None)

        # Changing color of the centre if the distance is too short
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Creating a Volume Bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Loading FPS and Put it on stream
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Video", img)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()

