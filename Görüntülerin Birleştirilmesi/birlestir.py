import cv2
import numpy as np

#resmi içe aktar
img=cv2.imread("lenna.png")
cv2.imshow("Orijinal",img)

#resmi yatay birleştir
hor=np.hstack((img,img))
cv2.imshow("Yatay",hor)

#resmi dikey birleştir
ver=np.vstack((img,img))
cv2.imshow("Dikey",ver)

cv2.waitKey(0)
cv2.destroyAllWindows()