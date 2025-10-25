# Tennis Performance Prediction

![Rezultate Model](example.png)

Acest script construiește un model de regresie liniară pentru a prezice câștigurile jucătorilor de tenis pe baza oportunităților de break point.

## Ce face scriptul:

1. **Încarcă datele** din fișierul `tennis_stats.csv`
2. **Construiește un model de predicție** care folosește `BreakPointsOpportunities` pentru a prezice `Winnings`
3. **Antrenează modelul** pe 80% din date și îl testează pe 20%
4. **Calculează R²** pentru a măsura calitatea predicțiilor
5. **Generează vizualizări**:
   - Plot dispesie Real vs. Prezis
   - Plot Feature vs. Outcome cu linia de regresie

Scriptul demonstră analiza predictivă în tenis, arătând relația dintre oportunitățile de break point și câștigurile financiare.
