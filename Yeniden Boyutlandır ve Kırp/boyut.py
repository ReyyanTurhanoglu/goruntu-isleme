import cv2
img=cv2.imread("lenna.png")
print("Resim boyutu: ",img.shape)
cv2.imshow("Orijinal",img)

#resized
imgResized=cv2.resize(img,(800,800))
print("Resized Img Shape: ",imgResized.shape)
cv2.imshow("Img Resized",imgResized)

#kırp
imgCropped=img[:200,:300] #height-width
cv2.imshow("Kirpik resim",imgCropped)

cv2.waitKey(0)           # Klavyeden herhangi bir tuşa basılana kadar bekler
cv2.destroyAllWindows()  # Tuşa basıldığında açılan tüm pencereleri kapatır