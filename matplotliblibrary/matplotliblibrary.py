import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel('personel_list.xlsx')
# print(df)
plt.figure(figsize=(12,6))
# plt.plot(df.isim,df.maas)
# plt.plot(df.isim,df.yas)
# plt.xlabel('İsim')
# plt.ylabel('Yaş')
# plt.title('Personel Yaş Grafiği')
# plt.show()

# plt.subplot(1,2,1)
# plt.plot(df.isim,df.maas)
# plt.subplot(1,2,2)
# plt.plot(df.isim,df.yas)
# plt.show()

# plt.scatter(df.isim,df.maas)
# plt.scatter(df.isim,df.yas)
# plt.show()

# plt.hist(df["maas"])
# plt.show()

# plt.hist(df["yas"])
# plt.show()

# plt.pie(df.maas,labels=df.isim)
# plt.pie(df.yas,labels=df.isim)
# plt.show()