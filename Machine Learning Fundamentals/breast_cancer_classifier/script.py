import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


breast_cancer_data = load_breast_cancer()

# print(breast_cancer_data.data[0])
# print(breast_cancer_data.feature_names)
# print(breast_cancer_data.target[0])
# print(breast_cancer_data.target_names[0])

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

# print(len(training_data))

# 14. Creează lista valorilor k (de la 1 la 100)
k_list = range(1, 101)

# 15. Creează lista goală pentru acurateți
accuracies = []

for k in k_list: # Poți folosi k_list sau range(1, 101) aici
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  score = (classifier.score(validation_data, validation_labels))
  # 15. În loc să printezi, adaugă acuratețea la lista 'accuracies'
  accuracies.append(score)

# # Linia de printare a scorului a fost înlăturată/comentată
# print(score)

# 16. Desenează graficul
plt.plot(k_list, accuracies)

# 17. Adaugă etichete și titlu
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")

# 16. Afișează graficul
plt.show()