import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# POSE LANDMARKER MODELİ
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_lite.task"
)


# --------------------------------------------------
# POSE LANDMARKER AYARLARI
# --------------------------------------------------

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# POSE DETECTOR OLUŞTUR
# --------------------------------------------------

pose = vision.PoseLandmarker.create_from_options(
    options
)


# --------------------------------------------------
# VİDEO
# --------------------------------------------------

cap = cv2.VideoCapture("video5.mp4")

# Kamera kullanmak istersen:
# cap = cv2.VideoCapture(0)


# --------------------------------------------------
# FPS
# --------------------------------------------------

pTime = 0

frame_timestamp_ms = 0


# --------------------------------------------------
# POSE CONNECTIONS
# --------------------------------------------------

connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    (9, 10),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),

    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32)
]


# --------------------------------------------------
# VİDEO DÖNGÜSÜ
# --------------------------------------------------

while True:

    success, img = cap.read()

    if not success:
        break


    # --------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------

    imgRGB = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------
    # MediaPipe Image
    # --------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )


    # --------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------

    frame_timestamp_ms += 33


    # --------------------------------------------------
    # POSE TESPİTİ
    # --------------------------------------------------

    results = pose.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # --------------------------------------------------
    # LANDMARKLAR VAR MI?
    # --------------------------------------------------

    if results.pose_landmarks:

        # İlk kişinin landmarkları
        landmarks = results.pose_landmarks[0]


        h, w, _ = img.shape


        # --------------------------------------------------
        # LANDMARK KOORDİNATLARI
        # --------------------------------------------------

        for id, lm in enumerate(landmarks):

            cx = int(lm.x * w)
            cy = int(lm.y * h)


            # --------------------------------------------------
            # ÖRNEK: 13 NUMARALI LANDMARK
            # --------------------------------------------------

            if id == 13:

                cv2.circle(
                    img,
                    (cx, cy),
                    5,
                    (255, 0, 0),
                    cv2.FILLED
                )


        # --------------------------------------------------
        # POSE ÇİZGİLERİ
        # --------------------------------------------------

        for start, end in connections:

            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)

            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)


            cv2.line(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # --------------------------------------------------
        # LANDMARK NOKTALARI
        # --------------------------------------------------

        for id, lm in enumerate(landmarks):

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            cv2.circle(
                img,
                (cx, cy),
                5,
                (0, 0, 255),
                cv2.FILLED
            )


    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    cTime = time.time()

    fps = 1 / (cTime - pTime)

    pTime = cTime


    cv2.putText(
        img,
        "FPS: " + str(int(fps)),
        (10, 65),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2
    )


    # --------------------------------------------------
    # GÖRÜNTÜ
    # --------------------------------------------------

    cv2.imshow(
        "Pose Estimation",
        img
    )


    # --------------------------------------------------
    # Q -> ÇIKIŞ
    # --------------------------------------------------

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# TEMİZLİK
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()

pose.close()