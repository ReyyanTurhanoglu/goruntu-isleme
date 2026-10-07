import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Kamera
# -----------------------------

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0


# -----------------------------
# El bağlantıları
# -----------------------------

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


while True:

    success, img = cap.read()

    if not success:
        print("Kamera görüntüsü alınamadı.")
        break

    # BGR -> RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # MediaPipe Image
    mpImage = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )

    # El algılama
    results = detector.detect(mpImage)

    # -----------------------------
    # El bulunduysa
    # -----------------------------

    if results.hand_landmarks:

        for handLms in results.hand_landmarks:

            h, w, c = img.shape

            lmList = []

            # -----------------------------
            # Landmark'ları al
            # -----------------------------

            for id, lm in enumerate(handLms):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lmList.append([id, cx, cy])

                # -----------------------------
                # id == 4 noktasını mavi çiz
                # -----------------------------

                if id == 4:

                    cv2.circle(
                        img,
                        (cx, cy),
                        9,
                        (255, 0, 0),
                        cv2.FILLED
                    )

            # -----------------------------
            # Landmark noktalarını çiz
            # -----------------------------

            for id, cx, cy in lmList:

                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    cv2.FILLED
                )

            # -----------------------------
            # El bağlantılarını çiz
            # -----------------------------

            for start, end in connections:

                x1 = lmList[start][1]
                y1 = lmList[start][2]

                x2 = lmList[end][1]
                y2 = lmList[end][2]

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


    # -----------------------------
    # FPS
    # -----------------------------

    cTime = time.time()

    if pTime != 0:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0

    pTime = cTime

    cv2.putText(
        img,
        "FPS: " + str(int(fps)),
        (10, 75),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        5
    )


    # -----------------------------
    # Kamerayı göster
    # -----------------------------

    cv2.imshow("img", img)

    # q ile çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Kapat
# -----------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()
