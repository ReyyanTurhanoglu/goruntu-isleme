import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. MEDIAPIPE POSE LANDMARKER
# =========================================================

base_options = python.BaseOptions(
    model_asset_path="pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = vision.PoseLandmarker.create_from_options(options)


# =========================================================
# 2. VIDEO
# =========================================================

cap = cv2.VideoCapture("video1.mp4")

# Önceki egzersiz için:
# cap = cv2.VideoCapture("video2.mp4")


fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

frame_index = 0


# =========================================================
# 3. GENEL DEĞİŞKENLER
# =========================================================

count = 0

# Push-up için:
stage = "up"


# =========================================================
# 4. ÖNCEKİ HIGH KNEES DEĞİŞKENLERİ
# =========================================================

# Önceki egzersizde kullanılmıştı.
# Şu anda aktif değiller.

# left_stage = False
# right_stage = False


# =========================================================
# 5. AÇI HESAPLAMA FONKSİYONU
# =========================================================

def findAngle(img, p1, p2, p3, lmList, draw=True):

    # p1 = birinci nokta
    # p2 = açı noktası
    # p3 = üçüncü nokta

    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    # Noktaları numpy array'e çevir
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])

    # Açı hesaplama
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        -
        np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(radians * 180.0 / np.pi)

    # Açıyı 0-180 arasına getir
    if angle > 180:
        angle = 360 - angle

    # Dirsek noktasını göster
    if draw:

        cv2.circle(
            img,
            (x2, y2),
            8,
            (255, 0, 0),
            cv2.FILLED
        )

    return angle


# =========================================================
# 6. VIDEO LOOP
# =========================================================

while True:

    success, img = cap.read()

    if not success:
        break

    frame_index += 1


    # =====================================================
    # 7. BGR -> RGB
    # =====================================================

    imgRGB = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # 8. MEDIAPIPE IMAGE
    # =====================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )


    # =====================================================
    # 9. TIMESTAMP
    # =====================================================

    timestamp_ms = int(
        frame_index * 1000 / fps
    )


    # =====================================================
    # 10. POSE DETECTION
    # =====================================================

    results = pose.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # =====================================================
    # 11. LANDMARK KONTROLÜ
    # =====================================================

    if results.pose_landmarks:

        landmarks = results.pose_landmarks[0]

        h, w, _ = img.shape

        lmList = []


        # =================================================
        # 12. 33 LANDMARK'I PIXEL KOORDİNATLARINA ÇEVİR
        # =================================================

        for id, lm in enumerate(landmarks):

            cx = int(lm.x * w)
            cy = int(lm.y * h)

            lmList.append([
                id,
                cx,
                cy
            ])


        # =================================================
        # 13. POSE İSKELETİNİ ÇİZ
        # =================================================

        connections = (
            vision.PoseLandmarksConnections.POSE_LANDMARKS
        )

        for connection in connections:

            start = connection.start
            end = connection.end

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


        # =================================================
        # 14. LANDMARK NOKTALARINI ÇİZ
        # =================================================

        for lm in lmList:

            id = lm[0]
            x = lm[1]
            y = lm[2]

            cv2.circle(
                img,
                (x, y),
                4,
                (0, 0, 255),
                cv2.FILLED
            )


        # =================================================
        # =================================================
        #
        #       AKTİF EGZERSİZ: PUSH-UP
        #
        # =================================================
        # =================================================


        # -------------------------------------------------
        # PUSH-UP LANDMARKLARI
        # -------------------------------------------------
        #
        # Sol omuz   = 11
        # Sol dirsek = 13
        # Sol bilek  = 15
        #
        # Sağ omuz   = 12
        # Sağ dirsek = 14
        # Sağ bilek  = 16
        # -------------------------------------------------


        # =================================================
        # 15. SOL KOL AÇISI
        # =================================================

        left_angle = findAngle(
            img,
            11,     # Sol omuz
            13,     # Sol dirsek
            15,     # Sol bilek
            lmList
        )


        # =================================================
        # 16. SAĞ KOL AÇISI
        # =================================================

        right_angle = findAngle(
            img,
            12,     # Sağ omuz
            14,     # Sağ dirsek
            16,     # Sağ bilek
            lmList
        )


        # =================================================
        # 17. ORTALAMA AÇI
        # =================================================

        avg_angle = (
            left_angle + right_angle
        ) / 2


        # =================================================
        # 18. PUSH-UP SAYMA
        # =================================================

        # Kişi aşağı inerken
        # dirsek açısı küçülür.

        if avg_angle < 125:

            stage = "down"


        # Kişi yukarı çıkarken
        # dirsek açısı büyür.

        if avg_angle > 150:

            if stage == "down":

                count += 1

                stage = "up"


        # =================================================
        # 19. SOL AÇI
        # =================================================

        cv2.putText(
            img,
            f"L: {int(left_angle)}",
            (20, 40),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            2
        )


        # =================================================
        # 20. SAĞ AÇI
        # =================================================

        cv2.putText(
            img,
            f"R: {int(right_angle)}",
            (20, 75),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            2
        )


        # =================================================
        # 21. ORTALAMA AÇI
        # =================================================

        cv2.putText(
            img,
            f"AVG: {int(avg_angle)}",
            (20, 110),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 255),
            2
        )


        # =================================================
        # 22. STAGE
        # =================================================

        cv2.putText(
            img,
            f"Stage: {stage}",
            (20, 150),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 255, 0),
            2
        )


        # =================================================
        # 23. COUNT
        # =================================================

        cv2.putText(
            img,
            f"Count: {count}",
            (20, 200),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0, 255, 0),
            3
        )


        # =================================================
        # =================================================
        #
        #      ÖNCEKİ EGZERSİZ: HIGH KNEES
        #
        #      ŞU ANDA AKTİF DEĞİL
        #
        #      AŞAĞIDAKİ KODLAR SADECE
        #      REFERANS OLARAK BIRAKILDI.
        #
        # =================================================
        # =================================================


        # -------------------------------------------------
        # HIGH KNEES LANDMARKLARI
        # -------------------------------------------------
        #
        # Sol kalça        = 23
        # Sol diz          = 25
        # Sol ayak bileği  = 27
        #
        # Sağ kalça        = 24
        # Sağ diz          = 26
        # Sağ ayak bileği  = 28
        # -------------------------------------------------


        # -------------------------------------------------
        # SOL DİZ KONTROLÜ
        # -------------------------------------------------

        # left_knee_y = lmList[25][2]
        # left_hip_y = lmList[23][2]

        # left_raised = left_knee_y < left_hip_y


        # -------------------------------------------------
        # SAĞ DİZ KONTROLÜ
        # -------------------------------------------------

        # right_knee_y = lmList[26][2]
        # right_hip_y = lmList[24][2]

        # right_raised = right_knee_y < right_hip_y


        # -------------------------------------------------
        # SOL DİZ SAYMA
        # -------------------------------------------------

        # if left_raised and not left_stage:

        #     count += 1

        #     left_stage = True


        # if not left_raised:

        #     left_stage = False


        # -------------------------------------------------
        # SAĞ DİZ SAYMA
        # -------------------------------------------------

        # if right_raised and not right_stage:

        #     count += 1

        #     right_stage = True


        # if not right_raised:

        #     right_stage = False


    # =====================================================
    # 24. GÖRÜNTÜYÜ GÖSTER
    # =====================================================

    cv2.imshow(
        "Push Up Counter",
        img
    )


    # =====================================================
    # 25. Q TUŞU İLE ÇIKIŞ
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# 26. TEMİZLE
# =========================================================

cap.release()

cv2.destroyAllWindows()