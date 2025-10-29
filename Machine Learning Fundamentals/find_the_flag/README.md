# Find the Flag - Machine Learning Classification

## Descriere

Acest script Python implementează un clasificator de arbori de decizie pentru a prezice continentul (Europa sau Oceania) pe baza caracteristicilor steagurilor naționale. Proiectul demonstrează concepte fundamentale de machine learning, inclusiv optimizarea hiperparametrilor și pruning-ul arborilor de decizie.

## Ce face scriptul

### 1. Încărcarea și Explorarea Datelor
- Încarcă dataset-ul de steaguri naționale de la UCI Machine Learning Repository
- Analizează distribuția țărilor pe continente
- Calculează valorile medii ale predictorilor pentru Europa și Oceania

### 2. Pregătirea Datelor
- Filtrează datele pentru a include doar țările din Europa (landmass=3) și Oceania (landmass=6)
- Creează variabile dummy pentru predictorii categorici
- Împarte datele în seturi de antrenare și testare (60% antrenare, 40% testare)

### 3. Optimizarea Hiperparametrilor
- Testează diferite valori pentru `max_depth` (1-20) pentru a găsi adâncimea optimă
- Vizualizează relația dintre adâncimea arborelui și acuratețea modelului
- Identifică adâncimea care oferă cea mai mare acuratețe

### 4. Pruning-ul Arborelui
- Aplică pruning prin cost complexity (CCP) pentru a preveni overfitting-ul
- Testează diferite valori pentru `ccp_alpha` (10^-3 până la 1)
- Găsește valoarea optimă pentru `ccp_alpha`

### 5. Vizualizarea Rezultatelor
- Afișează arborele de decizie final optimizat
- Prezintă grafice pentru analiza performanței modelului

## Caracteristici ale Steagurilor Analizate

Scriptul utilizează următoarele caracteristici ale steagurilor:
- **Culori**: roșu, verde, albastru, auriu, alb, negru, portocaliu
- **Design**: dungi, bare, cercuri, cruci, sărituri, sferturi
- **Elemente**: stele, semilună, triunghiuri, icoane
- **Aspecte vizuale**: culorile principale, animații, text

## Rezultate Așteptate

Scriptul va afișa:
1. Distribuția țărilor pe continente
2. Valorile medii ale caracteristicilor pentru Europa și Oceania
3. Grafice de performanță pentru diferite adâncimi
4. Cea mai bună acuratețe și adâncimea corespunzătoare
5. Grafice de performanță pentru pruning
6. Cea mai bună acuratețe cu pruning și valoarea ccp_alpha
7. Arborele de decizie final optimizat

## Dependențe

```python
import codecademylib3
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
```

## Utilizare

1. Asigură-te că ai instalate toate dependențele necesare
2. Rulează scriptul: `python script.py`
3. Observă graficele și rezultatele afișate în consolă

## Concepte de Machine Learning Demonstrate

- **Clasificare cu arbori de decizie**
- **Optimizarea hiperparametrilor** (max_depth)
- **Pruning pentru prevenirea overfitting-ului**
- **Validare prin împărțirea datelor**
- **Vizualizarea arborilor de decizie**
- **Analiza performanței modelului**

## Dataset

Datele provin din UCI Machine Learning Repository:
- **Sursa**: https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
- **Conținut**: Caracteristici ale steagurilor naționale din întreaga lume
- **Scop**: Clasificarea continentului pe baza caracteristicilor vizuale ale steagului
