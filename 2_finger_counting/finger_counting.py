import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# MediaPipe Hand Landmarker
# --------------------------------------------------

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


# --------------------------------------------------
# Kamera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)


# --------------------------------------------------
# İki nokta arasındaki mesafe
# --------------------------------------------------

def distance(p1, p2):

    return math.sqrt(
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


# --------------------------------------------------
# 3 nokta arasındaki açı
# --------------------------------------------------

def calculate_angle(a, b, c):

    # a -> b
    ba = (
        a[1] - b[1],
        a[2] - b[2]
    )

    # c -> b
    bc = (
        c[1] - b[1],
        c[2] - b[2]
    )

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine_angle = dot_product / (
        magnitude_ba * magnitude_bc
    )

    # -1 ile 1 arasında tut
    cosine_angle = max(
        -1,
        min(1, cosine_angle)
    )

    angle = math.degrees(
        math.acos(cosine_angle)
    )

    return angle


# --------------------------------------------------
# Parmak sayma
# --------------------------------------------------

def count_fingers(lmList):

    fingers = []

    # --------------------------------------------------
    # BAŞPARMAK
    # --------------------------------------------------

    # Thumb:
    # 2 = MCP
    # 3 = IP
    # 4 = TIP

    thumb_angle = calculate_angle(
        lmList[2],
        lmList[3],
        lmList[4]
    )

    if thumb_angle > 150:
        fingers.append(1)
    else:
        fingers.append(0)


    # --------------------------------------------------
    # INDEX
    # --------------------------------------------------

    index_angle = calculate_angle(
        lmList[5],
        lmList[6],
        lmList[7]
    )

    if index_angle > 150:
        fingers.append(1)
    else:
        fingers.append(0)


    # --------------------------------------------------
    # MIDDLE
    # --------------------------------------------------

    middle_angle = calculate_angle(
        lmList[9],
        lmList[10],
        lmList[11]
    )

    if middle_angle > 150:
        fingers.append(1)
    else:
        fingers.append(0)


    # --------------------------------------------------
    # RING
    # --------------------------------------------------

    ring_angle = calculate_angle(
        lmList[13],
        lmList[14],
        lmList[15]
    )

    if ring_angle > 150:
        fingers.append(1)
    else:
        fingers.append(0)


    # --------------------------------------------------
    # PINKY
    # --------------------------------------------------

    pinky_angle = calculate_angle(
        lmList[17],
        lmList[18],
        lmList[19]
    )

    if pinky_angle > 150:
        fingers.append(1)
    else:
        fingers.append(0)


    return fingers.count(1), fingers


# --------------------------------------------------
# Kamera döngüsü
# --------------------------------------------------

while True:

    success, img = cap.read()

    if not success:
        print("Kamera görüntüsü alınamadı.")
        break


    # --------------------------------------------------
    # Ayna görüntüsü
    # --------------------------------------------------

    img = cv2.flip(img, 1)


    # --------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------

    imgRGB = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )


    # --------------------------------------------------
    # El tespiti
    # --------------------------------------------------

    results = detector.detect(mp_image)


    total_fingers = 0


    # --------------------------------------------------
    # Eller bulunduysa
    # --------------------------------------------------

    if results.hand_landmarks:

        h, w, _ = img.shape


        for hand_index, hand_landmarks in enumerate(
            results.hand_landmarks
        ):

            lmList = []


            # --------------------------------------------------
            # Landmark koordinatları
            # --------------------------------------------------

            for id, lm in enumerate(hand_landmarks):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lmList.append(
                    [id, cx, cy]
                )


            # --------------------------------------------------
            # El etiketi
            # --------------------------------------------------

            hand_label = results.handedness[
                hand_index
            ][0].category_name


            # --------------------------------------------------
            # El bağlantıları
            # --------------------------------------------------

            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (0, 17)
            ]


            # --------------------------------------------------
            # Çizgiler
            # --------------------------------------------------

            for start, end in connections:

                cv2.line(
                    img,
                    (
                        lmList[start][1],
                        lmList[start][2]
                    ),
                    (
                        lmList[end][1],
                        lmList[end][2]
                    ),
                    (0, 255, 0),
                    2
                )


            # --------------------------------------------------
            # Landmark noktaları
            # --------------------------------------------------

            for id, cx, cy in lmList:

                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    cv2.FILLED
                )


            # --------------------------------------------------
            # Parmak say
            # --------------------------------------------------

            finger_count, fingers = count_fingers(
                lmList
            )

            total_fingers += finger_count


            # --------------------------------------------------
            # El bilgisi
            # --------------------------------------------------

            x = lmList[0][1]
            y = lmList[0][2]


            cv2.putText(
                img,
                f"{hand_label}: {finger_count}",
                (x - 50, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )


        # --------------------------------------------------
        # Toplam
        # --------------------------------------------------

        cv2.putText(
            img,
            f"Toplam: {total_fingers}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            4
        )


    # --------------------------------------------------
    # Kamera
    # --------------------------------------------------

    cv2.imshow(
        "Two Hand Finger Counting",
        img
    )


    # q -> çıkış
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Temizlik
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()

detector.close()