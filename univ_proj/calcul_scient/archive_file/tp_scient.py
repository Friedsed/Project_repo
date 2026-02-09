import time

# paramètres
N_min = 10
N_max = 80
step = 10
tol = 1e-6

# accumulateurs de temps
time_interp_lagr = 0.0
time_interp_newt = 0.0
time_jac_lagr = 0.0
time_jac_newt = 0.0
time_gs_lagr = 0.0
time_gs_newt = 0.0

# pour comparer éventuellement au cas "direct N=80"
time_jac_direct = 0.0
time_gs_direct = 0.0

# pour stocker solutions et maillages successifs
T_prev = None
x_prev = None

for N in range(N_min, N_max + 1, step):
    print(f"\n=== Maillage N = {N} ===")
    A, b, x = build_systeme(N)

    if N == N_min:
        # départ champ nul
        T0 = np.zeros_like(b)

        # Jacobi direct
        t0 = time.perf_counter()
        T_jac, it_jac = jacobi(A, b, T0, tol=tol)
        t1 = time.perf_counter()
        time_jac_direct += (t1 - t0)

        # GS direct
        t0 = time.perf_counter()
        T_gs, it_gs = gauss_seidel(A, b, T0, tol=tol)
        t1 = time.perf_counter()
        time_gs_direct += (t1 - t0)

        print(f"Jacobi N={N} init 0 : {it_jac} itérations")
        print(f"GS     N={N} init 0 : {it_gs} itérations")

        # on conserve la solution convergée comme "coarse" pour N=20
        T_prev = T_jac.copy()  # ou T_gs, peu importe ici
        x_prev = x.copy()

    else:
        # interpolation du maillage précédent (x_prev, T_prev) vers x (N courant)

        # --- interpolation Newton ---
        t0 = time.perf_counter()
        a_newt = newton_coeff(x_prev, T_prev)
        T_init_newt = newton_eval(x_prev, a_newt, x)
        t1 = time.perf_counter()
        time_interp_newt += (t1 - t0)

        # --- interpolation Lagrange ---
        t0 = time.perf_counter()
        T_init_lagr = lagrange_interp(x_prev, T_prev, x)
        t1 = time.perf_counter()
        time_interp_lagr += (t1 - t0)

        # --- Jacobi avec init Newton ---
        t0 = time.perf_counter()
        T_jac_newt, it_jac_newt = jacobi(A, b, T_init_newt, tol=tol)
        t1 = time.perf_counter()
        time_jac_newt += (t1 - t0)

        # --- Jacobi avec init Lagrange ---
        t0 = time.perf_counter()
        T_jac_lagr, it_jac_lagr = jacobi(A, b, T_init_lagr, tol=tol)
        t1 = time.perf_counter()
        time_jac_lagr += (t1 - t0)

        # --- GS avec init Newton ---
        t0 = time.perf_counter()
        T_gs_newt, it_gs_newt = gauss_seidel(A, b, T_init_newt, tol=tol)
        t1 = time.perf_counter()
        time_gs_newt += (t1 - t0)

        # --- GS avec init Lagrange ---
        t0 = time.perf_counter()
        T_gs_lagr, it_gs_lagr = gauss_seidel(A, b, T_init_lagr, tol=tol)
        t1 = time.perf_counter()
        time_gs_lagr += (t1 - t0)

        print(f"Jacobi N={N} Newton   : {it_jac_newt} itérations")
        print(f"Jacobi N={N} Lagrange : {it_jac_lagr} itérations")
        print(f"GS     N={N} Newton   : {it_gs_newt} itérations")
        print(f"GS     N={N} Lagrange : {it_gs_lagr} itérations")

        # on choisit par exemple la solution GS-Newton comme nouvelle "coarse"
        T_prev = T_gs_newt.copy()
        x_prev = x.copy()

# affichage des temps totaux
print("\n=== Temps totaux ===")
print(f"Temps interpolation Newton   : {time_interp_newt:.6f} s")
print(f"Temps interpolation Lagrange : {time_interp_lagr:.6f} s")
print(f"Temps Jacobi (init Newton)   : {time_jac_newt:.6f} s")
print(f"Temps Jacobi (init Lagrange) : {time_jac_lagr:.6f} s")
print(f"Temps GS (init Newton)       : {time_gs_newt:.6f} s")
print(f"Temps GS (init Lagrange)     : {time_gs_lagr:.6f} s")
