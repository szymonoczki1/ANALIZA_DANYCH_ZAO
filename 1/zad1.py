import numpy as np

# macierz hilberta nxn, + 1 we wzorze bo indeksy od 0
def create_hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H

def main():
    n = 20
    
    A = create_hilbert_matrix(n)
    print(f"Macierz Hilberta {n}×{n}:")
    print(A)
    print()
    
    # axis=1 -> po wierszach
    b = np.sum(A, axis=1)
    print("Wektor prawej strony b (suma elementów w wierszach):")
    print(b)
    print()
    
    # metoda 1
    x1 = np.linalg.solve(A, b)
    print("Rozwiązanie metodą numpy.linalg.solve:")
    print(x1)
    print()
    
    # metoda 2
    x2, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("Rozwiązanie metodą numpy.linalg.lstsq:")
    print(x2)
    print()
    
    # metoda 3
    A_inv = np.linalg.inv(A)
    x3 = A_inv.dot(b)
    print("Rozwiązanie metodą inv(A).dot(b):")
    print(x3)
    print()
    

    print("PORÓWNANIE ROZWIĄZAŃ:")
    print()
    
    # roznice wzgledem solve (metoda 1)
    diff_lstsq = np.linalg.norm(x1 - x2)
    diff_inv = np.linalg.norm(x1 - x3)
    
    print(f"Norma różnicy (solve vs lstsq): {diff_lstsq}")
    print(f"Norma różnicy (solve vs inv): {diff_inv}")
    print()
    
    # A.dot(x0) mnozenie macierzy przez wektor x || A.dot(x) - b == Ax - b
    residuum1 = np.linalg.norm(A.dot(x1) - b)
    residuum2 = np.linalg.norm(A.dot(x2) - b)
    residuum3 = np.linalg.norm(A.dot(x3) - b)
    
    print("Norma residuum |Ax - b|₂:")
    print(f"  solve:  {residuum1}")
    print(f"  lstsq:  {residuum2}")
    print(f"  inv:    {residuum3}")
    print()
    
    # wskaźnik uwarunkowania
    cond_A = np.linalg.cond(A)
    print(f"Wskaźnik uwarunkowania macierzy cond(A): {cond_A:.2e}")
    print()
    
if __name__ == "__main__":
    main()
