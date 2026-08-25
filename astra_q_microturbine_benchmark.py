import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETRI E SIMULAZIONE SISTEMA DINAMICO
# ==========================================
class MicroTurbineSimulator:
    """
    Simulatore 0D ridotto della microturbina basato sui dati 
    della tesi di M. L. Erario (GTM 140 / JetCat P140 Rxi-B).
    Stato x = [omega (krpm), EGT (°C), W_f (g/s)]
    Controllo u = [dot_m_f, P_load]
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        # Parametri nominali del rotore e termici
        self.omega_max = 125.0  # krpm
        self.inertia = 0.001    # inerzia ridotta
        
    def step(self, x, u, degradation_factor=1.0):
        omega, egt, wf = x
        dot_m_f, p_load = u
        
        # Degradazione applicata a efficienza compressore/turbina che influenza la spinta e la EGT
        eff_deg = degradation_factor
        
        # Dinamica non lineare accoppiata (approssimazione del modello GSP)
        d_omega = (1.0 / self.inertia) * (25.0 * dot_m_f * eff_deg - 0.15 * omega - 0.05 * p_load)
        d_egt = 12.0 * dot_m_f - 0.08 * (omega * eff_deg) + 0.02 * egt
        d_wf = 5.0 * (dot_m_f - wf) # DINAMICA attuatore/sistema combustione
        
        # Aggiornamento dello stato con Eulero
        omega_next = np.clip(omega + d_omega * self.dt, 30.0, 130.0)
        egt_next = np.clip(egt + d_egt * self.dt, 400.0, 800.0)
        wf_next = np.clip(wf + d_wf * self.dt, 1.0, 10.0)
        
        return np.array([omega_next, egt_next, wf_next])

    def generate_trajectory(self, steps, u_profile, degradation_factor=1.0):
        trajectory = np.zeros((steps, 3))
        # Stato iniziale tipico (idle stabilizzato)
        x = np.array([80.0, 550.0, 3.0]) 
        for t in range(steps):
            trajectory[t] = x
            u = u_profile[t]
            x = self.step(x, u, degradation_factor)
        return trajectory

# ==========================================
# 2. LIFTING DI KOOPMAN EDMD (ASTRA-Q CORE)
# ==========================================
class KoopmanEDMD:
    """
    Costruisce lo spazio delle osservabili Psi(x) e calcola 
    l'operatore di Koopman K per via algebrica (EDMD).
    """
    def __init__(self, polynomial_degree=2):
        self.deg = polynomial_degree
        self.K = None

    def _lift(self, X):
        # X shape: (N_samples, state_dim)
        n_samples, dim = X.shape
        lifted = [X]
        # Inseriamo osservabili polinomiali non lineari (es. interazioni e quadrati)
        if self.deg >= 2:
            quads = [X[:, i:i+1] * X[:, j:j+1] for i in range(dim) for j in range(i, dim)]
            lifted.append(np.hstack(quads))
        return np.hstack(lifted)

    def fit(self, X, X_next):
        Psi_X = self._lift(X)
        Psi_X_next = self._lift(X_next)
        
        # Risoluzione ai minimi quadrati per K: Psi_X_next = Psi_X * K^T  =>  K = (Psi_X^\dagger * Psi_X_next)^T
        # Usiamo pseudo-inversa di Moore-Penrose
        Psi_X_pinv = np.linalg.pinv(Psi_X)
        self.K = (Psi_X_pinv @ Psi_X_next).T
        return self.K

    def predict(self, x_current):
        psi_x = self._lift(x_current.reshape(1, -1))
        psi_next = (self.K @ psi_x.T).T
        # Ritorna allo spazio originale prendendo le prime componenti dello stato
        return psi_next[:, :3]


# ==========================================
# 3. ESECUZIONE DEL BENCHMARK E CONFRONTO
# ==========================================
if __name__ == "__main__":
    print("=== ASTRA-Q MicroTurbine Benchmark Iniziato ===")
    
    sim = MicroTurbineSimulator(dt=0.1)
    
    # Generazione profilo di controllo di test (escursione di manetta e carico)
    steps = 1500
    t_array = np.linspace(0, steps * 0.1, steps)
    # Profilo di fuel flow e carico variabile nel tempo
    u_profile = np.zeros((steps, 2))
    u_profile[:, 0] = 3.0 + 1.5 * np.sin(0.05 * t_array) + 0.5 * np.cos(0.2 * t_array) # dot_m_f
    u_profile[:, 1] = 10.0 + 2.0 * np.sin(0.02 * t_array) # P_load

    # 1. Generazione traiettoria HEALTHY (baseline)
    traj_healthy = sim.generate_trajectory(steps, u_profile, degradation_factor=1.0)
    
    # 2. Generazione traiettoria DEGRADED (es. usura compressore/turbina del 15%, d = 0.15)
    degradation_level = 0.85 
    traj_degraded = sim.generate_trajectory(steps, u_profile, degradation_factor=degradation_level)

    # Preparazione dati per EDMD di Koopman
    X_train = traj_healthy[:-1]
    X_next_train = traj_healthy[1:]

    koopman = KoopmanEDMD(polynomial_degree=2)
    K_healthy = koopman.fit(X_train, X_next_train)

    # Adattiamo un operatore anche sullo stato degradato per il confronto spettrale
    koopman_deg = KoopmanEDMD(polynomial_degree=2)
    K_degraded = koopman_deg.fit(traj_degraded[:-1], traj_degraded[1:])

    # Estrazione autovalori (Spettro dei modi dinamici)
    evals_healthy, _ = np.linalg.eig(K_healthy)
    evals_degraded, _ = np.linalg.eig(K_degraded)

    # Calcolo indicatore di salute basato sullo shift spettrale (D_K o variazione autovalori)
    spectral_shift = np.linalg.norm(np.sort(np.abs(evals_degraded)) - np.sort(np.abs(evals_healthy)))

    print(f"\n[Risultati Analisi Spettrale ASTRA-Q]:")
    print(f" - Variazione spettrale norma autovalori (Health Indicator): {spectral_shift:.5f}")
    print(f" - Condizione motore: Degradato al {(1.0 - degradation_level)*100:.1f}%")

    # Verifica della legge di perturbazione e scaling non additivo (I_nonadd ~ epsilon^p)
    epsilons = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    non_add_indices = []
    
    for eps in epsilons:
        traj_pert = sim.generate_trajectory(200, u_profile[:200], degradation_factor=(1.0 - eps))
        # Differenza rispetto all'healthy in norma L2 normalizzata
        diff = np.linalg.norm(traj_pert - traj_healthy[:200]) / np.linalg.norm(traj_healthy[:200])
        non_add_indices.append(diff)

regression_coeffs = np.polyfit(np.log(epsilons), np.log(non_add_indices), 1)
p_estimated = regression_coeffs[0]

print(f"\n[Verifica Scaling Legge di Perturbazione]:")
print(f" - Esponente stimato p empirico: {p_estimated:.4f} (Target teorico ~ 2.0)")
print("=== Benchmark Completato con Successo ===")
