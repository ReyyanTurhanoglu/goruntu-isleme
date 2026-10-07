import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# FACE LANDMARKER
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

faceLandmarker = vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------
# VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture("video3.mp4")

fps_video = cap.get(cv2.CAP_PROP_FPS)

frame_index = 0

pTime = 0


# --------------------------------------------------
# LOOP
# --------------------------------------------------

while True:

    success, img = cap.read()

    if not success:
        break


    # BGR -> RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # OpenCV görüntüsünü MediaPipe görüntüsüne çevir
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imgRGB
    )


    # --------------------------------------------------
    # FACE LANDMARK DETECTION
    # --------------------------------------------------

    timestamp_ms = int(frame_index * 1000 / fps_video)

    results = faceLandmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    frame_index += 1


    # --------------------------------------------------
    # FACE LANDMARKS
    # --------------------------------------------------

    if results.face_landmarks:

        for faceLms in results.face_landmarks:

            h, w, _ = img.shape


            # --------------------------------------------------
            # YÜZ ÜÇGENLERİNİ ÇİZ
            # --------------------------------------------------

            for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:

                start = connection.start
                end = connection.end

                x1 = int(faceLms[start].x * w)
                y1 = int(faceLms[start].y * h)

                x2 = int(faceLms[end].x * w)
                y2 = int(faceLms[end].y * h)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )


            # --------------------------------------------------
            # LANDMARK KOORDİNATLARI
            # --------------------------------------------------

            for id, lm in enumerate(faceLms):

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                print([id, cx, cy])


    # --------------------------------------------------
    # FPS
    # --------------------------------------------------

    cTime = time.time()

    if cTime != pTime:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0

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
    # GÖRÜNTÜYÜ GÖSTER
    # --------------------------------------------------

    cv2.imshow("Face Mesh", img)


    # Q ile çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# KAPAT
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()