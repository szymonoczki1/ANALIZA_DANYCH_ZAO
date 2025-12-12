import numpy as np
import matplotlib.pyplot as plt

def generate_clean_data(x):
    return np.sin(2 * np.pi * x)

def generate_noisy_data(x, sigma=0.12):
    y_clean = generate_clean_data(x)
    noise = np.random.normal(0, sigma, size=len(x))
    return y_clean + noise

def calculate_rmse(y_true, y_pred):
    # np array pozwalaja na odejmowanie wektorów bez pętli
    return np.sqrt(np.mean((y_true - y_pred)**2))

def create_plot(x_fine, y_clean_fine, x_nodes, y_noisy, polynomials, rmse_values, sigma):
    degrees = [1, 2, 3]
    colors = ['red', 'green', 'blue']
    
    plt.figure(figsize=(12, 8))
    
    # og krzywa sin(2πx)
    plt.plot(x_fine, y_clean_fine, 'k-', linewidth=2, label='Oryginalna: y = sin(2πx)', alpha=0.7)
    
    # nody z szumem
    plt.scatter(x_nodes, y_noisy, color='black', s=80, zorder=5, 
                label=f'Dane z szumem (σ={sigma})', marker='o', edgecolors='white', linewidths=1.5)
    
    # dodanie wielomianow
    for i, degree in enumerate(degrees):
        y_fit = polynomials[degree](x_fine)
        plt.plot(x_fine, y_fit, color=colors[i], linewidth=2, 
                label=f'Wielomian stopnia {degree} (RMSE={rmse_values[degree]:.4f})', 
                linestyle='--', alpha=0.8)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Dopasowanie wielomianów do danych z szumem', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(844)

    n_points = 15
    x_nodes = np.linspace(0, 1, n_points)
    
    # dane do wykresu
    x_fine = np.linspace(0, 1, 200)
    y_clean_fine = generate_clean_data(x_fine)
    
    # dane z szumem
    sigma = 0.12
    y_noisy = generate_noisy_data(x_nodes, sigma)
    y_clean_nodes = generate_clean_data(x_nodes)
    

    print("DOPASOWANIE WIELOMIANÓW METODĄ NAJMNIEJSZYCH KWADRATÓW")
    print()
    print(f"Liczba węzłów: {n_points}")
    print(f"Odchylenie standardowe szumu: {sigma}")
    print()
    
    degrees = [1, 2, 3]
    polynomials = {}
    rmse_values = {}
    
    for degree in degrees:
        # dopasowanie wielomianu
        coeffs = np.polyfit(x_nodes, y_noisy, degree)
        poly = np.poly1d(coeffs)
        #print(poly)
        polynomials[degree] = poly
        
        # obliczanie rmse
        y_pred = poly(x_nodes)
        rmse = calculate_rmse(y_clean_nodes, y_pred)
        rmse_values[degree] = rmse
        
        print(f"\nStopień {degree}:")
        print(f"  Współczynniki: {coeffs}")
        print(f"  RMSE: {rmse:.6f}")
        print()

    create_plot(x_fine, y_clean_fine, x_nodes, y_noisy, polynomials, rmse_values, sigma)

if __name__ == "__main__":
    main()
