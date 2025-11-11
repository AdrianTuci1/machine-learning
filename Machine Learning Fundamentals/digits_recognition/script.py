import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

model = KMeans(n_clusters=10, random_state=42)

model.fit(digits.data)

fig = plt.figure(figsize=(8,3))

fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):

  ax = fig.add_subplot(2, 5, 1 + i)

  ax.imshow(model.cluster_centers_[i].reshape((8,8))),
  cmap=plt.cm.binary


plt.show()


new_samples = np.array([
[0.15,4.81,7.55,7.62,7.62,6.71,1.83,0.00,2.75,7.62,4.35,2.67,3.20,7.24,3.81,0.00,4.19,7.02,0.00,0.00,0.00,7.02,3.81,0.00,5.72,5.49,0.00,0.00,1.45,7.62,2.75,0.00,2.44,1.68,0.00,0.00,4.20,7.32,0.46,0.00,0.00,0.00,0.00,1.30,7.32,4.58,0.00,0.00,0.15,4.27,5.64,7.55,7.62,5.41,4.58,3.74,0.30,5.80,6.10,6.10,6.10,6.10,6.33,7.40],
[0.00,0.46,4.35,7.47,7.62,6.79,1.22,0.00,0.00,4.35,7.55,4.50,3.13,7.02,4.35,0.00,1.53,7.63,4.42,0.00,0.00,5.80,5.57,0.00,3.59,7.47,0.31,0.00,0.00,4.73,6.41,0.00,5.03,6.33,0.00,0.00,0.00,3.89,6.86,0.00,5.34,5.49,0.00,0.00,0.15,5.41,6.79,0.00,4.35,7.47,4.42,2.59,6.03,7.62,3.36,0.00,0.38,4.12,7.47,7.62,7.02,2.37,0.00,0.00],
[0.08,6.10,7.62,7.62,7.55,4.73,0.00,0.00,2.06,7.62,3.89,2.82,4.88,7.63,0.53,0.00,5.03,6.71,0.15,0.00,3.66,7.47,0.38,0.00,2.59,2.59,0.00,0.00,6.03,5.64,0.00,0.00,0.00,0.00,0.00,0.99,7.63,3.28,0.00,0.00,0.00,0.00,0.00,4.66,7.47,0.46,0.00,0.00,0.99,4.20,5.87,7.62,7.24,6.10,6.10,6.02,4.19,7.62,6.79,4.88,4.57,4.57,4.57,5.11],
[3.05,4.57,5.34,6.10,5.80,6.86,7.09,0.53,4.27,6.10,5.80,5.26,5.34,6.10,7.63,1.07,0.00,0.00,0.00,0.00,0.00,6.26,5.72,0.00,0.23,2.21,2.59,2.29,3.81,7.62,2.90,0.00,2.75,7.62,7.62,7.62,7.62,7.62,0.99,0.00,0.15,2.14,5.26,5.19,7.10,5.80,0.00,0.00,0.00,0.00,0.00,1.83,7.62,2.59,0.00,0.00,0.00,0.00,0.00,1.07,4.96,0.23,0.00,0.00]
])

new_labels = model.predict(new_samples)

print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

