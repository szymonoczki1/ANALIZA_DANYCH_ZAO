import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)


# Krok 1: Wczytaj lub wygeneruj dane i wyświetl pierwsze 10 wierszy


n_samples = 100

# zwraca liste 100 liczb z przedzialu 0-30
temp = np.random.randint(0, 31, size=n_samples)
# zwraca liste 100 liczb z przedzialu -10 do 10
epsilon = np.random.randint(-10, 11, size=n_samples)
# wzor z pdf, tez lista 100 elementow
sales = 5.0 * temp + 50 + epsilon

# tworzymy pandas DataFrame, 100 wierszy
data = pd.DataFrame({
    'temperatura': temp,
    'sprzedaz': sales
})

print("Pierwsze 10 wierszy danych:")
print(data.head(10))

data.to_csv('ice_cream_sales_data.csv', index=False)


# Krok 2: Podziel dane na zbiór treningowy (80%) i testowy (20%)


# dataframe 2d dla X, series 1d dla y, bo sklearn tak wymaga
X = data[['temperatura']]
y = data['sprzedaz']

# dzielimy dane na treningowe i testowe 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Krok 3: Dopasuj model regresji liniowej (OLS): sales ~ temp


# tworzymy i dopasowujemy model na danych treningowych
model = LinearRegression()
model.fit(X_train, y_train)

# pobieramy parametry modelu
slope = model.coef_[0]
intercept = model.intercept_

print(f"\nParametry modelu:")
print(f"  Współczynnik kierunkowy (slope): {slope:.4f}")
print(f"  Wyraz wolny (intercept): {intercept:.4f}")


# Krok 4: Oblicz MSE i R² na zbiorze testowym


# przewidujemy wartosci y dla X_test
y_pred_test = model.predict(X_test)

# na ich podstawie liczymy MSE i R^2
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"\nMetryki na zbiorze testowym:")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")


# Krok 5: Narysuj wykres punktowy (temp, sales) oraz wykreśl prostą dopasowaną na zbiorze treningowym
       

plt.figure(figsize=(10, 6))

# Wykres punktowy danych treningowych
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Dane treningowe', s=50)

# Prosta dopasowana
temp_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
sales_pred = model.predict(temp_range)
plt.plot(temp_range, sales_pred, color='red', linewidth=2, 
         label=f'Model: y = {slope:.2f}x + {intercept:.2f}')

plt.xlabel('Temperatura (°C)', fontsize=11)
plt.ylabel('Sprzedaż (liczba gałek)', fontsize=11)
plt.title('Regresja liniowa: sprzedaż lodów vs temperatura', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('wykres_regresji.png', dpi=300, bbox_inches='tight')
plt.show()


# Krok 6: Krótko zinterpretuj wyniki


print("\nInterpretacja wyników:")
print(f"""
Współczynnik kierunkowy (slope = {slope:.4f}):
  Wzrost temperatury o 1°C prowadzi do wzrostu sprzedaży o ~{slope:.2f} gałek lodów.
  
Wyraz wolny (intercept = {intercept:.4f}):
  Teoretyczna sprzedaż przy temperaturze 0°C wynosi ~{intercept:.1f} gałek.
  
Sensowność modelu:
  Model jest sensowny - wyższa temperatura naturalnie zwiększa sprzedaż lodów.
  R² = {r2:.4f} oznacza, że {r2*100:.1f}% zmienności sprzedaży jest wyjaśnione
  przez temperaturę, co wskazuje na silną zależność liniową.
""")

# Zapisanie wyników do pliku
with open('wyniki_analizy.txt', 'w', encoding='utf-8') as f:
    f.write("WYNIKI ANALIZY REGRESJI\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Parametry modelu:\n")
    f.write(f"  Slope: {slope:.4f}\n")
    f.write(f"  Intercept: {intercept:.4f}\n\n")
    f.write(f"Metryki (zbiór testowy):\n")
    f.write(f"  MSE: {mse:.4f}\n")
    f.write(f"  R²: {r2:.4f}\n\n")
    f.write(f"Interpretacja:\n")
    f.write(f"  Wzrost temperatury o 1°C zwiększa sprzedaż o {slope:.2f} gałek.\n")
    f.write(f"  Model wyjaśnia {r2*100:.1f}% zmienności sprzedaży.\n")
    f.write(f"  Model jest sensowny - temperatura jest dobrym predyktorem sprzedaży lodów.\n")
