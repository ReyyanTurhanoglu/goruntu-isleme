# 📷 OpenCV ile Görüntü İşleme Çalışmaları

Bu repository, görüntü işleme alanındaki kişisel çalışma ve uygulamalarımdan oluşmaktadır. İçerik, Udemy üzerinde **DATAI TEAM** tarafından hazırlanan **"Python OpenCV ile Sıfırdan Uzmanlığa Görüntü İşleme"** kursu kapsamında gerçekleştirilen pratikleri, ödevleri ve gerçek zamanlı bilgisayarlı görü projelerini kapsamaktadır.

---

## 📂 İçerik ve Klasör Yapısı

Depo; veri analitiği temelleri, temel görüntü işleme teknikleri ve MediaPipe destekli ileri seviye tespit/takip projeleri olmak üzere modüler olarak düzenlenmiştir:

### 📊 1. Temel Kütüphaneler ve Veri Analitiği
* **`numpylibrary.py`**: Matris, dizi işlemleri ve piksel manipülasyonu temelleri.
* **`pandaslibrary/`**: Veri analizi ve veri çerçevesi (DataFrame) kullanımı.
* **`matplotliblibrary/`**: Görüntü ve grafiklerin görselleştirilmesi.

---

### 🛠️ 2. Temel Görüntü İşleme Operasyonları
* **`resmi_içe_aktarma/`**: Görsel okuma, görüntüleme ve kaydetme işlemleri.
* **`video_içe_aktarma/`**: Video dosyalarını içeri aktarma ve kare kare işleme.
* **`KameraAçma ve VideoKaydı/`**: Web kamerasından anlık akış alma ve video formatında diske kaydetme.
* **`Yeniden Boyutlandır ve Kırp/`**: Yeniden ölçeklendirme (resizing) ve ROI (Region of Interest) kırpma.
* **`Şekiller ve Metin/`**: Görüntü üzerine çizgi, dikdörtgen, daire çizimi ve metin ekleme.
* **`Görüntülerin Birleştirilmesi/`**: Görselleri dikey/yatay eksende birleştirme.
* **`Görüntüleri Karıştırmak/`**: Farklı ağırlıklarla pikselleri harmanlama (image blending).
* **`Perspektif Çarpıtma/`**: Açı düzeltme ve 4 noktalı perspektif dönüşümleri (warp perspective).
* **`Bulanıklaştırma/`**: Gürültü giderme, Gauss, Medyan ve Ortalama filtreleme.
* **`morfoloji/`**: Erozyon, genişletme (dilation), açılma ve kapanma morfolojik operatörleri.
* **`gradyanlar/`**: Sobel ve Laplacian filtreleri ile kenar tespiti.
* **`histogram/`**: Piksel yoğunluğu histogramları ve histogram eşitleme.
* **`görüntü eşikleme/`**: Eşikleme (thresholding), adaptif eşikleme ve ikili (binary) görüntüleme.
* **`ödev1/`**: Temel aşamayı pekiştiren kapsamlı çalışma.

---

### 🚀 3. İleri Düzey Uygulamalar ve Projeler
Kurs boyunca geliştirilen gerçek zamanlı ve model tabanlı projeler:

1. **`1_hand_tracking/`**: Gerçek zamanlı el ve eklem noktaları takibi.
2. **`2_finger_counting/`**: El anatomisi üzerinden anlık parmak sayma algoritması.
3. **`3_pose_estimation/`**: Vücut iskelet yapısı ve duruş kestirimi.
4. **`4_personal_trainer/`**: Egzersiz tekrarı sayımı ve eklem açısı analizi yapan sanal antrenör.
5. **`5_face_detection/`**: Yüz tespiti uygulaması.
6. **`6_face_mesh/`**: Yüz üzerinde 468 referans noktasını haritalandıran Face Mesh modeli.
7. **`7_parking_space_counter/`**: Otoparktaki boş ve dolu park yerlerini tespit edip sayan sistem.
8. **`8_road_line_detection/`**: Otonom sürüş sistemlerinde kullanılan şerit çizgisi tespiti.
9. **`9_sleep_detection/`**: Göz kapanma süresini takip ederek uyku/yorgunluk durumu tespiti (Drowsiness Detection).

---

## 💻 Kullanılan Teknolojiler ve Kütüphaneler

* **Python 3.x**
* **OpenCV (`cv2`)**: Görüntü ve video işleme
* **MediaPipe**: El, yüz ve duruş modelleri
* **NumPy**: Matris ve sayısal hesaplamalar
* **Matplotlib**: Görselleştirme
* **Pandas**: Veri manipülasyonu

---

## ⚙️ Kurulum ve Çalıştırma

Projeleri yerel ortamınızda çalıştırmak için:

1. **Repoyu klonlayın:**
   ```bash
   git clone [https://github.com/kullanici-adi/repo-adi.git](https://github.com/kullanici-adi/repo-adi.git)
   cd repo-adi
