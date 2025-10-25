import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Incarcă datele
stats = pd.read_csv('tennis_stats.csv')
outcome = stats[['Winnings']]

# --- Model 1.1: BPO -> Winnings (Cerința 4) ---
features_bpo = stats[['BreakPointsOpportunities']]

# Split pentru train/test
X_train_bpo, X_test_bpo, Y_train_bpo, Y_test_bpo = train_test_split(
    features_bpo, outcome, train_size=0.8, random_state=42
)
model_bpo = LinearRegression()
model_bpo.fit(X_train_bpo, Y_train_bpo)
r2_bpo = model_bpo.score(X_test_bpo, Y_test_bpo)
prediction_bpo = model_bpo.predict(X_test_bpo)

# --- Vizualizare Model Unifactorial cu Linia de Regresie ---

plt.figure(figsize=(10, 6))

# 1. Puncte de dispersie (Datele de testare)
# Rețineți: Acesta este un plot Real vs. Prezis, NU Feature vs. Outcome.
plt.scatter(Y_test_bpo, prediction_bpo, alpha=0.6, color='darkgreen', label='Predicții (Test Set)')

# 2. Linia Ideală (Y=X)
max_val = max(Y_test_bpo.max().iloc[0], prediction_bpo.max())
min_val = min(Y_test_bpo.min().iloc[0], prediction_bpo.min())
plt.plot([min_val, max_val], [min_val, max_val], 
         color='red', linestyle='--', linewidth=2, label='Linia Ideală (Real = Prezis)')

# 3. Linia de Regresie (Prezice Winnings pe baza BPO-ului din setul de testare)
# Pentru a desena linia de regresie, trebuie să plotăm: BPO (X_test) vs. Predicție (prediction_bpo)
# Totuși, cerința specifică un plot al 'predicted outcome against the actual outcome'.
# Desenăm o linie de regresie pe plot-ul BPO vs. Winnings pentru claritate:
# Reutilizăm X_test_bpo pentru a desena linia de regresie în spațiul FEATURĂ vs. OUTCOME.

# --- VIZUALIZARE ALTERNATIVĂ (Feature vs. Outcome) care arată LINIA DE REGRESIE direct ---
plt.figure(figsize=(10, 6))

# Punctele de dispersie (Setul de testare)
plt.scatter(X_test_bpo, Y_test_bpo, alpha=0.6, color='darkgreen', label='Date Reale (Test)')

# Linia de Regresie: Folosim aceleași BPO-uri de test, dar plotăm predicțiile
# Aceasta este linia care arată cum funcționează modelul nostru.
plt.plot(X_test_bpo, prediction_bpo, 
         color='orange', linewidth=3, label='Linia de Regresie Prezisă') 

plt.title(f'2. Model Unifactorial: BPO vs. Winnings (R²: {r2_bpo:.4f})')
plt.xlabel('Break Points Opportunities (Test Set)')
plt.ylabel('Winnings ($)')
plt.legend()
plt.show()

# --- NOTĂ: Vizualizarea corectă pentru Cerința 4 (Reale vs. Prezise) nu necesită linia de regresie! ---
# Cerința 4 cere plt.scatter(outcome_test, prediction), care este un plot de evaluare.
# Pe acest plot se folosește 'Linia Ideală' (Y=X), care a fost deja inclusă în pasul anterior.
