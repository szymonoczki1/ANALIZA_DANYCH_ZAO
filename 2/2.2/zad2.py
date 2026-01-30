import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# 1. Generowanie danych
n_samples = 300

# uniform generuje floaty
area = np.random.uniform(40, 200, n_samples)
people = np.random.randint(1, 6, n_samples)
windows = np.random.randint(1, 11, n_samples)
age = np.random.uniform(0, 80, n_samples)

epsilon = np.random.normal(0, 20, n_samples)
consumption = 0.8 * area + 10 * people + 2 * windows - 0.1 * age + epsilon

df = pd.DataFrame({
    'area': area,
    'people': people,
    'windows': windows,
    'age': age,
    'consumption': consumption
})

df.to_csv('energy.csv', index=False)
print("Dane zapisane do energy.csv")

# 2. Podział na zbiór treningowy i testowy (80/20)
X = df[['area', 'people', 'windows', 'age']]
y = df['consumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Dopasowanie regresji liniowej wielowymiarowej (OLS)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Obliczanie MSE i R² na zbiorze testowym
y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"\nMSE na zbiorze testowym: {mse:.2f}")
print(f"R² na zbiorze testowym: {r2:.4f}")

# 5. Współczynniki modelu i ich interpretacja
print(f"\nWspółczynniki modelu:")
print(f"Wyraz wolny: {model.intercept_:.4f}")

feature_names = ['area', 'people', 'windows', 'age']
coefficients = model.coef_

for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

print(f"\nInterpretacja:")
print(f"area: wzrost o 1 m² zwiększa zużycie o {coefficients[0]:.4f} kWh")
print(f"people: wzrost o 1 osobę zwiększa zużycie o {coefficients[1]:.4f} kWh")
print(f"windows: wzrost o 1 okno zwiększa zużycie o {coefficients[2]:.4f} kWh")
print(f"age: wzrost o 1 rok zmienia zużycie o {coefficients[3]:.4f} kWh")

# 6. Standaryzacja cech i ponowne dopasowanie modelu

# StandardScaler automatycznie przekształca dane do średniej 0 i wariancji 1
# musimy przeksztalcic dane w taki sposob zeby srednia = 0 a srednia(x^2) = 1
# dzieki temu bedzie nam latwiej porownac jaki wplyw maja dane zmienne na wynik, bo beda w tej samej skali
# np jak area po standaryzacji bedzie miala wspolczynnik 39.5 to znaczy ze wzrost o 1 std dev area zwieksza zuzycie o 39.5 kWh
# gdzie std dev to jak bardzo dane area sa rozproszone wokol sredniej, w taki sam sposob jak dla innych cech
scaler = StandardScaler()

# fit oblicza srednia i std dev na danych treningowych
# transform przeksztalca dane tak zeby srednia = 0 a std dev = 1 -> wariancja = std dev^2 == 1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
# fit dopasowuje model na standaryzowanych danych treningowych
model_scaled.fit(X_train_scaled, y_train)

coefficients_scaled = model_scaled.coef_

# po standaryzacji intercept zmienia sie na srednia wartosc y w danych treningowych
print(f"\nWspółczynniki po standaryzacji:")
print(f"Wyraz wolny: {model_scaled.intercept_:.4f}")
for name, coef in zip(feature_names, coefficients_scaled):
    print(f"{name}: {coef:.4f}")

print(f"\nPorównanie współczynników:")
for i, name in enumerate(feature_names):
    print(f"{name}: bez standaryzacji={coefficients[i]:.4f}, ze standaryzacją={coefficients_scaled[i]:.4f}")
