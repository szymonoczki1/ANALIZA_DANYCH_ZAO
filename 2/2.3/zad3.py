import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

# 1. Wczytaj dane
df = pd.read_csv('clients.csv')
features = ["age", "annual_spending", "visits_per_month", "avg_basket_value"]
X = df[features].values

# 2. Standaryzacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Klastrowanie k-średnich
# mozemy zwiekszyc n_init w celu uzyskania lepszych wynikow
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# fit tworzy centroidy jako srednie punktow
# predict przypisuje kazdy punkt do najblizszego centroidu
cluster_labels = kmeans.fit_predict(X_scaled)

# 4. Centra klastrów w oryginalnych jednostkach
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

print("Centra klastrów (oryginalne jednostki):")
for i in range(3):
    print(f"Klaster {i}:")
    for j, feature in enumerate(features):
        print(f"  {feature}: {centers_original[i][j]:.2f}")

# 5. Miary jakości
# Inertia -> suma kwadratów odległości punktów od najblizszych centroidów
inertia = kmeans.inertia_
print(f"\nInertia: {inertia:.2f}")

# Silhouette score -> miara jakości klastrowania (jak podobne sa obiekty do ich klastrów), wartości od -1 do 1, 1 to najlepsza jakość
silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Zgodność z true_segment
mapping = {}
for i in range(3):
    #mask = [True, False, True, False, True]
    mask = cluster_labels == i
    if mask.sum() > 0:
        most_common = mode(df.loc[mask, 'true_segment'], keepdims=True)[0][0]
        mapping[i] = most_common

mapped_labels = np.array([mapping[label] for label in cluster_labels])
accuracy = (mapped_labels == df['true_segment'].values).mean()
print(f"Zgodność z prawdziwymi segmentami: {accuracy*100:.2f}%")

# 6. Within-Cluster Sum of Squares (WSS) w oryginalnej skali
wss_original = 0
for i in range(len(X)):
    cluster_id = cluster_labels[i]
    center = centers_original[cluster_id]
    diff = X[i] - center
    wss_original += np.sum(diff ** 2)

print(f"WSS (oryginalna skala): {wss_original:.2f}")

# WSS w skali znormalizowanej, nie rozumiem czemu robimy to w orginalnej skali, roznica pomiedzy wydatkami a np. wizytami jest olbrzymia
wss_scaled = 0
for i in range(len(X_scaled)):
    cluster_id = cluster_labels[i]
    center = centers_scaled[cluster_id]
    diff = X_scaled[i] - center
    wss_scaled += np.sum(diff ** 2)

print(f"WSS (skala znormalizowana): {wss_scaled:.2f}")

# 7. k-NN - Przypisanie nowego klienta
knn = KNeighborsClassifier(n_neighbors=5)
# trenujemy klasyfikator
knn.fit(X_scaled, cluster_labels)

new_customer = {
    "age": 33,
    "annual_spending": 2900,
    "visits_per_month": 5,
    "avg_basket_value": 360
}

new_customer_array = np.array([[
    new_customer["age"],
    new_customer["annual_spending"],
    new_customer["visits_per_month"],
    new_customer["avg_basket_value"]
]])

new_customer_scaled = scaler.transform(new_customer_array)
predicted_cluster = knn.predict(new_customer_scaled)[0]

print(f"\nNowy klient - przewidywany klaster: {predicted_cluster}")

# Centroid przypisanego klastra
assigned_centroid_original = centers_original[predicted_cluster]
print(f"Centroid klastra {predicted_cluster} (oryginalne jednostki):")
for i, feature in enumerate(features):
    print(f"  {feature}: {assigned_centroid_original[i]:.2f}")

# Odległość do centroidu
assigned_centroid_scaled = centers_scaled[predicted_cluster]
distance_scaled = np.linalg.norm(new_customer_scaled[0] - assigned_centroid_scaled)
distance_original = np.linalg.norm(new_customer_array[0] - assigned_centroid_original)

print(f"Odległość do centroidu (skala znormalizowana): {distance_scaled:.4f}")
print(f"Odległość do centroidu (oryginalna skala): {distance_original:.2f}")