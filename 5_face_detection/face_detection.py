import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# 1. FACE DETECTOR MODELİ
# =========================================================

base_options = python.BaseOptions(
    model_asset_path="blaze_face_short_range.tflite"
)


# =========================================================
# 2. FACE DETECTION AYARLARI
# =========================================================

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.20
)


# =========================================================
# 3. FACE DETECTOR OLUŞTUR
# =========================================================

faceDetection = vision.FaceDetector.create_from_options(
    options
)


# =========================================================
# 4. VIDEO
# =========================================================

cap = cv2.VideoCapture("video3.mp4")


# =========================================================
# 5. VIDEO LOOP
# =========================================================

while True:

    success, img = cap.read()

    # Video bittiyse döngüden çık
    if not success:
        break


    # =====================================================
    # 6. BGR -> RGB
    # =====================================================

    imgRGB = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # 7. MEDIAPIPE IMAGE
    # =====================================================

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )


    # =====================================================
    # 8. FACE DETECTION
    # =====================================================

    results = faceDetection.detect(
        mp_image
    )


    # =====================================================
    # 9. YÜZLER BULUNDUYSA
    # =====================================================

    if results.detections:

        for id, detection in enumerate(
            results.detections
        ):

            # ---------------------------------------------
            # Bounding Box bilgilerini al
            # ---------------------------------------------

            bbox = detection.bounding_box


            # ---------------------------------------------
            # Bounding Box koordinatları
            # ---------------------------------------------

            x = bbox.origin_x
            y = bbox.origin_y

            width = bbox.width
            height = bbox.height


            # ---------------------------------------------
            # Dikdörtgen çiz
            # ---------------------------------------------

            cv2.rectangle(
                img,
                (x, y),
                (x + width, y + height),
                (0, 255, 255),
                2
            )


            # ---------------------------------------------
            # Yüz ID'sini göster
            # ---------------------------------------------

            cv2.putText(
                img,
                f"Face: {id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1.5,
                (0, 255, 255),
                2
            )


    # =====================================================
    # 10. GÖRÜNTÜYÜ GÖSTER
    # =====================================================

    cv2.imshow(
        "Face Detection",
        img
    )


    # =====================================================
    # 11. Q TUŞU İLE ÇIKIŞ
    # =====================================================

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


# =========================================================
# 12. TEMİZLE
# =========================================================

cap.release()

cv2.destroyAllWindows()