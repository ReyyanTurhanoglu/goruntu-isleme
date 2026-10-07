import cv2
import matplotlib.pyplot as plt

# Karıştırma
img1 = cv2.imread("img1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread("img2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Boyutlandırma
img1 = cv2.resize(img1, (600, 600))
img2 = cv2.resize(img2, (600, 600))

# Karıştırma (Blending)
blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)

# Ekrana basma
plt.figure()
plt.imshow(img1)
plt.title("Resim 1")

plt.figure()
plt.imshow(img2)
plt.title("Resim 2")

plt.figure()
plt.imshow(blended)
plt.title("Karistirilmis (Blended)")

# Tüm Matplotlib pencerelerini ekrana getiren komut:
plt.show()