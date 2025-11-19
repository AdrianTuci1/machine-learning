import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz


fig, ax = plt.subplots()

print(aaron_judge.columns)
print(aaron_judge.description.unique())

aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})
print(aaron_judge.type.unique())

print(aaron_judge['plate_x'])

aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])

plt.scatter(x = aaron_judge['plate_x'], y = aaron_judge['plate_z'], c = aaron_judge['type'], cmap = plt.cm.coolwarm, alpha = 0.25)

training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

# --- OPTIMIZAREA HIPERPARAMETRILOR (Cerința Aditivă) ---

best_score = 0
best_gamma = 0
best_C = 0

# Testarea unei game de valori pentru Gamma și C (Puteri ale lui 10)
gammas_to_test = [4, 3, 1, 100]
Cs_to_test = [0.1, 1, 10, 100]

print("--- Începe Optimizarea Hiperparametrului ---")


for gamma in gammas_to_test:
    for C in Cs_to_test:

        classifier_tuned = SVC(kernel='rbf', gamma=gamma, C=C)
        classifier_tuned.fit(training_set[['plate_x', 'plate_z']], training_set['type'])


        score = classifier_tuned.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])

        print(f"Gamma: {gamma}, C: {C}, Acuratețe: {score:.4f}")

        # 3. Urmărirea celui mai bun scor
        if score > best_score:
            best_score = score
            best_gamma = gamma
            best_C = C

print("--- Optimizarea S-a Finalizat ---")
print(f"🏆 Cel Mai Bun Scor pe Setul de Validare: {best_score:.4f}")
print(f"Parametrii Optimi: Gamma = {best_gamma}, C = {best_C}")

classifier = SVC(kernel = 'rbf', gamma = 100, C = 100)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

print(classifier.score(training_set[['plate_x', 'plate_z']], training_set['type']))

ax.set_ylim(-2, 2)
draw_boundary(ax, classifier)

plt.show()


