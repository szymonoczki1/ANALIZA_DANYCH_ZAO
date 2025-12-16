import numpy as np
from scipy import linalg
import time

def create_symmetric_tridiagonal_matrix(n, diagonal_value=2.0, off_diagonal_value=-1.0):
    d = np.full(n, diagonal_value)  # diagonal
    a = np.full(n-1, off_diagonal_value)  # off-diagonal
    
    A = np.diag(d) + np.diag(a, k=1) + np.diag(a, k=-1)
    
    return A

def main():
    n = 100
    diagonal_value = 2.0
    off_diagonal_value = -1.0
    
    A = create_symmetric_tridiagonal_matrix(n, diagonal_value, off_diagonal_value)
    
    print(f"Rozmiar macierzy: {n}×{n}")
    print(f"Wartość na diagonali głównej: {diagonal_value}")
    print(f"Wartość na diagonalach pobocznych: {off_diagonal_value}")
    print()
    
    print(A)
    print()
    
    is_symmetric = np.allclose(A, A.T)
    print(f"Macierz jest symetryczna: {is_symmetric}")
    print()
    
    print("METODA 1: scipy.linalg.eig (dla ogólnych macierzy)")
    print()
    
    start_time = time.perf_counter()
    eigenvalues_eig, eigenvectors_eig = linalg.eig(A)
    time_eig = time.perf_counter() - start_time
    
    # eig zawsze zwraca typ liczb complex128 wiec musimmy wziąć część rzeczywistą
    eigenvalues_eig = np.real(eigenvalues_eig)
    # eigvalsh auto sortuje wiec musimy posortowac ta zeby moc porownac
    eigenvalues_eig_sorted = np.sort(eigenvalues_eig)
    
    print(f"Czas wykonania: {time_eig:.6f} sekund")
    print(f"Pierwsze 5 wartości własnych: {eigenvalues_eig_sorted[:5]}")
    print(f"Ostatnie 5 wartości własnych: {eigenvalues_eig_sorted[-5:]}")
    print()
    

    print("METODA 2: scipy.linalg.eigvalsh (dla macierzy symetrycznych/hermitowskich)")
    print()
    
    start_time = time.perf_counter()
    eigenvalues_eigvalsh = linalg.eigvalsh(A)
    time_eigvalsh = time.perf_counter() - start_time
    
    print(f"Czas wykonania: {time_eigvalsh:.6f} sekund")
    print(f"Pierwsze 5 wartości własnych: {eigenvalues_eigvalsh[:5]}")
    print(f"Ostatnie 5 wartości własnych: {eigenvalues_eigvalsh[-5:]}")
    print()
    

    print("PORÓWNANIE WYNIKÓW")
    print()
    

    difference = np.abs(eigenvalues_eig_sorted - eigenvalues_eigvalsh)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)
    
    print(f"Maksymalna różnica między metodami: {max_diff:.20f}")
    print(f"Średnia różnica między metodami:    {mean_diff:.20f}")
    print()
    
    print("Porównanie czasów wykonania:")
    print(f"  scipy.linalg.eig:      {time_eig:.6f} s")
    print(f"  scipy.linalg.eigvalsh: {time_eigvalsh:.6f} s")
    print(f"  Przyspieszenie: {time_eig/time_eigvalsh:.2f}x")
    print()


if __name__ == "__main__":
    main()