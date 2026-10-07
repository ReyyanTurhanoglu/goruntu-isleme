import cv2

# resmi içe aktarma
img = cv2.imread("fotograf1.jpeg",0) # 0 parametresi resmi gri tonlamalı olarak yükler

#görselleştirme
cv2.imshow("Gri Tonlamalı Resim",img)
k=cv2.waitKey(0) & 0xFF # herhangi bir tuşa basılmasını bekler
if k==27: # ESC tuşuna basılırsa
    cv2.destroyAllWindows() # tüm pencereleri kapatır
elif k==ord('s'): # 's' tuşuna basılırsa
    cv2.imwrite("fotograf1_gri.jpeg",img) # resmi kaydeder
    cv2.destroyAllWindows() # tüm pencereleri kapatır