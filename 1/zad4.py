import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def polynomial(x):
    return x**4 - 3*x**3 + 2*x - 1

def find_sign_change_intervals(func, x_range, n_points=1000):
    x_min, x_max = x_range
    x_values = np.linspace(x_min, x_max, n_points)
    y_values = func(x_values)
    
    intervals = []
    for i in range(len(y_values) - 1):
        if y_values[i] * y_values[i+1] < 0:
            intervals.append((x_values[i], x_values[i+1]))
    
    return intervals

def create_plot(roots):
    x = np.linspace(-5, 5, 1000)
    plt.plot(x, polynomial(x), 'b-')
    plt.axhline(0, color='k', linestyle='--')
    if roots:
        plt.plot(roots, [0]*len(roots), 'ro')
    plt.grid()
    plt.show()

def main():
    print("ZNAJDOWANIE PIERWIASTKÓW WIELOMIANU")
    print("Wielomian: p(x) = x^4 - 3x^3 + 2x - 1")
    print()
    
    search_range = (-5, 5)
    intervals = find_sign_change_intervals(polynomial, search_range, n_points=2000)
    
    print(f"Znaleziono {len(intervals)} przedziały ze zmianą znaku:")
    for i, (a, b) in enumerate(intervals, 1):
        print(f"  Przedział {i}: [{a:.4f}, {b:.4f}]")
    print()
    


    print("Użycie metody brenta")
    print()
    
    roots = []
    for i, (a, b) in enumerate(intervals, 1):
        try:
            root = brentq(polynomial, a, b)
            roots.append(root)
            
            # sprawdzenie
            p_root = polynomial(root)
            
            print(f"Pierwiastek {i}:")
            print(f"  x = {root:.10f}")
            print(f"  p(x) = {p_root:.10f} (sprawdzenie)")
            print()
        except ValueError as e:
            print(e)
    
    create_plot(roots)


if __name__ == "__main__":
    main()
