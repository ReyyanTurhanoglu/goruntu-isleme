import cv2

video_name="MOT17-04-DPM.mp4"
cap=cv2.VideoCapture(video_name)
print("Genişlik: ",cap.get(3))
print("Yükseklik ",cap.get(4))

if cap.isOpened()==False:
    print("Hata")

while True:
    ret,frame=cap.read()
    if ret==True:
        
        cv2.imshow("Video",frame)
    else:
        break
    if cv2.waitKey(10) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

