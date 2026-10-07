# opencv kütüphanesini içe aktaralım
import cv2
import numpy as np

# matplotlib kütüphanesini içe aktaralım
import matplotlib.pyplot as plt

# resmi siyah beyaz (gri tonlamalı) olarak içe aktaralım
# (Resmin dosya adının 'odev1.jpg' olduğunu varsayıyoruz)
img = cv2.imread("odev1.jpg", 0)

# resmi çizdirelim
plt.figure()
plt.imshow(img, cmap="gray")
plt.title("Orijinal Gri Resim")
plt.axis("off")
plt.show()

# resmin boyutuna bakalım
print("Resmin Boyutu (Yükseklik, Genişlik):", img.shape)

# resmi 4/5 oranında yeniden boyutlandıralım ve resmi çizdirelim
# fx ve fy parametreleri yatay ve dikey ölçeklendirme çarpanlarıdır
img_resized = cv2.resize(img, (0, 0), fx=4/5, fy=4/5)
print("Yeniden Boyutlandırılmış Boyut:", img_resized.shape)

plt.figure()
plt.imshow(img_resized, cmap="gray")
plt.title("4/5 Oraninda Yeniden Boyutlandirilmis")
plt.axis("off")
plt.show()

# orijinal resme bir yazı ekleyelim mesela "kopek" ve resmi çizdirelim
# Orijinal matrisi bozmamak için kopyasını alıyoruz
img_text = img.copy()
cv2.putText(
    img_text, 
    text="kopek", 
    org=(430, 150),                    # Yazının başlangıç (x, y) koordinatı (köpeğin civarı)
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale=1.2, 
    color=255,                         # Gri resimde 255 = beyaz yazı
    thickness=3
)

plt.figure()
plt.imshow(img_text, cmap="gray")
plt.title("Yazi Eklenmis Resim")
plt.axis("off")
plt.show()

# orijinal resmin 50 threshold değeri üzerindekileri beyaz yap altındakileri siyah yapalım,
# binary threshold yöntemi kullanalım ve resmi çizdirelim
_, thresh_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(thresh_img, cmap="gray")
plt.title("Binary Threshold (Eşik = 50)")
plt.axis("off")
plt.show()

# orijinal resme gaussian bulanıklaştırma uygulayalım ve resmi çizdirelim
# ksize tek sayılardan oluşmalıdır (örn: (7, 7)), sigmaX=0 standart sapmayı otomatik hesaplar
blurred_img = cv2.GaussianBlur(img, (7, 7), sigmaX=0)

plt.figure()
plt.imshow(blurred_img, cmap="gray")
plt.title("Gaussian Bulaniklastirma")
plt.axis("off")
plt.show()

# orijinal resme Laplacian gradyan uygulayalım ve resmi çizdirelim
# Kenar geçişlerindeki negatif değerleri kaybetmemek için cv2.CV_64F kullanılır, ardından mutlak değer alınır
laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

plt.figure()
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Gradyan (Kenarlar)")
plt.axis("off")
plt.show()

# orijinal resmin histogramını çizdirelim
hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

plt.figure()
plt.plot(hist, color="black")
plt.title("Gri Seviye Histogrami")
plt.xlabel("Piksel Degeri (0 - 255)")
plt.ylabel("Piksel Sayisi (Frekans)")
plt.xlim([0, 256])
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()