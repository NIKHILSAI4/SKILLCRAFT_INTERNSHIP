import cv2
import mediapipe as mp

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tipIds = [4, 8, 12, 16, 20]

while True:

    success, img = cap.read()

    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    totalFingers = 0
    gesture = ""

    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            mpDraw.draw_landmarks(
                img,
                handLms,
                mpHands.HAND_CONNECTIONS
            )

            lmList = []

            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lmList.append([id, cx, cy])

            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers
            for id in range(1,5):

                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)

            if totalFingers == 0:
                gesture = "Fist"

            elif totalFingers == 1:
                gesture = "One Finger"

            elif totalFingers == 2:
                gesture = "Two Fingers"

            elif totalFingers == 3:
                gesture = "Three Fingers"

            elif totalFingers == 4:
                gesture = "Four Fingers"

            elif totalFingers == 5:
                gesture = "Open Palm"

            # Bounding box

            xList = [pt[1] for pt in lmList]
            yList = [pt[2] for pt in lmList]

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)

            cv2.rectangle(
                img,
                (xmin-20, ymin-20),
                (xmax+20, ymax+20),
                (255,0,255),
                2
            )

    cv2.putText(
        img,
        f"Gesture: {gesture}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()