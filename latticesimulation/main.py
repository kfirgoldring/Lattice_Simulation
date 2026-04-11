import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from pathlib import Path
import scipy.sparse.linalg as sla
from latticesimulation.lattice_simulation import LatticeSimulation

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
VORTICES_Y_GAUGE_DIR = PLOTS_DIR / "vortices" / "y_gauge"


def _show_or_close():
    # Avoid UserWarning on non-interactive backends (e.g., FigureCanvasAgg).
    if "agg" in str(plt.get_backend()).lower():
        plt.close()
    else:
        plt.show()


def run_simulation():
    # --- Parameters ---
    Lx, Ly = 30, 30  # Lattice dimensions
    t_hop = 1.0  # Hopping parameter (t = 2*lambda in your image)
    mass_values=[1] # Different mass values to test

    # Run simulation for both Boundary Conditions
    bcs = ['pbc', 'obc']
    results = {}
    for bc in bcs:
        print(f"Running simulation for {bc.upper()}...")
        sim = LatticeSimulation(Lx, Ly, boundary_condition=bc)
        results[bc] = {}

        for mass in mass_values:
            print(f"  Mass = {mass}")
            # Dirac
            H_dir = sim.get_dirac_hamiltonian(t=t_hop, m=mass)
            H_sch=sim.get_schrodinger_hamiltonian(t=t_hop)
            energies = sim.get_spectrum(H_dir)#eigenvalues of the hamiltonian
            real_dos=(1/(8*np.pi*t_hop**2))*np.abs(energies)#theoretical density of states for 2D Dirac
            #print(real_dos)
            window = 2.0 * abs(mass)#keep energies within this window -2m<e<2m
            #window=0.5
            filtered_energies = energies[(energies >= -window) & (energies <= window)]
            results[bc][mass] = filtered_energies
            #results[bc][mass] = energies
    # --- Plotting Dirac Results for All Masses ---
    print("Plotting Dirac results for all masses...")
    fig, (ax_pbc, ax_obc) = plt.subplots(1, 2, figsize=(16, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    # PBC plot
    for i, mass in enumerate(mass_values):
        n,edges,patches=ax_pbc.hist(results['pbc'][mass], bins=50, color=colors[i],
                   alpha=0.6, label=f'm={mass}', edgecolor='black', linewidth=0.5)

    ax_pbc.set_title(f"Dirac PBC -m={mass} ,Lattice sites: {Lx}x{Ly}", fontsize=14)
    dE=edges[1]-edges[0]#bin width
    ax_pbc.set_ylabel("DOS (Count)", fontsize=12)
    ax_pbc.set_xlabel("Energy (E)", fontsize=12)
    ax_pbc.legend()
    ax_pbc.grid(True, alpha=0.3)
    real_dos=real_dos*dE*Lx*Ly#scale theoretical dos by bin width
    ax_pbc.plot(energies, real_dos, color='black', label='Theoretical DOS')
    ax_pbc.legend()
    # OBC plot
    for i, mass in enumerate(mass_values):
        ax_obc.hist(results['obc'][mass], bins=50, color=colors[i],
                   alpha=0.6, label=f'm={mass}', edgecolor='black', linewidth=0.5)

    ax_obc.set_title(f"Dirac OBC - Multiple Masses\nLattice: {Lx}x{Ly}", fontsize=14)
    ax_obc.set_ylabel("DOS (Count)", fontsize=12)
    ax_obc.set_xlabel("Energy (E)", fontsize=12)
    ax_obc.plot(energies, real_dos, color='black', label='Theoretical DOS')
    ax_obc.legend()
    ax_obc.grid(True, alpha=0.3)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / "dirac_real_vs_numerical.jpg", dpi=300)
    _show_or_close()

def run_bcs_simulation(Lx,Ly,t_hop,mass,mu,Delta0,bc):
    '''
    plot bcs DOS in both PBC and OBC in real space
    2D Wilson–Dirac model with s-wave pairing

    '''
    print(f"Running BCS simulation for {bc.upper()}...")
    sim = LatticeSimulation(Lx, Ly, boundary_condition=bc)#sim object
    H_bdg = sim.get_bdg_josephson_hamiltonian(t=t_hop, m=mass, mu=mu, Delta0=Delta0,phi_josephson=0,L1=0,L2=Lx)#get hamiltonian
    energies_bdg = sim.get_spectrum(H_bdg)#energy eigenalues
    # --- Plotting BdG Results ---
    fig, ax_pbc = plt.subplots(1, 1, figsize=(16, 8))
    # PBC plot
    ax_pbc.hist(energies_bdg, bins=50, color='blue',
               alpha=0.6, edgecolor='black', linewidth=0.5)
    ax_pbc.set_title(f"BdG PBC\nLattice: {Lx}x{Ly} ,Delta={Delta0}", fontsize=14)
    ax_pbc.set_ylabel("DOS (Count)", fontsize=12)
    ax_pbc.set_xlabel("Energy (E)", fontsize=12)
    ax_pbc.grid(True, alpha=0.3)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "bdg_pbc.jpg", dpi=300)
    _show_or_close()
    print(f"  Completed BCS simulation for {bc.upper()}.")
    return energies_bdg

def run_josephson_current_operator(Lx, Ly, t_hop,alpha, mass, mu, Delta0, L1, L2,
                                  x_cut=None, eps_pos=1e-10):
    """
    Compute Josephson current I(φ) in 2 ways: the bond-current operator expectation value, and
    using josephson relation I=-DeltaE/DeltaPhi.
    alpha=phi0/(a^2 B) flux per plaquette in units of phi0
    The junction is defined between x=L1 and x=L2 (inclusive).

    In the operator method We sum currents across a cut: (x_cut,y) -> (x_cut+1,y) for all y.
    Choose x_cut inside the weak link for a clean "junction current".
    """
    print(f"Running Josephson current-operator simulation")

    sim = LatticeSimulation(Lx=Lx, Ly=Ly, boundary_condition_x='obc', boundary_condition_y='pbc')

    # Choose a cut inside the junction by default (middle of [L1, L2])
    if x_cut is None:
        x_cut = (L1 + L2) // 2
        x_cut = max(0, min(Lx - 2, x_cut))  # ensure valid

    phi_list = np.linspace(0, 2*np.pi, 21, endpoint=False)
    current_list = []

    # Normal-state Hamiltonian (electron sector), used to extract hopping blocks T_ij
    H0 = sim.get_dirac_hamiltonian(t=t_hop, m=mass,alpha=alpha).tocsr()

    dim_e = 2 * sim.N  # electron-sector dimension (2 internal comps per site)
    energy_list=[]

    for phi in phi_list:
        # Build BdG for this phase
        H_bdg = sim.get_bdg_josephson_hamiltonian(
            t=t_hop, m=mass, mu=mu, Delta0=Delta0,
            phi_0=phi,alpha=alpha, L1=L1, L2=L2
        )
        evals, evecs = sim.get_spectrum(H_bdg)
        ground_state_energy=np.sum(evals[evals>0])#sum of negative energy states
        energy_list.append(ground_state_energy)
        # Select positive-energy eigenvectors (avoid numerical ~0)
        pos = evals > eps_pos

        # Split BdG eigenvectors into u (electron) and v (hole)
        U = evecs[:dim_e, :][:, pos]      # shape: (2N, #pos)
        V = evecs[dim_e:, :][:, pos]      # shape: (2N, #pos)

        # Electron correlator C = <c^\dagger c> = V V^\dagger  (T=0)
        C = V @ V.conj().T                # shape: (2N, 2N)

        # Sum bond currents across the cut
        I_tot = 0.0
        for y in range(Ly):
            i_site = x_cut + y * Lx
            j_site = (x_cut + 1) + y * Lx

            si = 2 * i_site  # starting index in electron sector
            sj = 2 * j_site

            # 2x2 hopping block from i->j in H0
            T_ij = H0[si:si+2, sj:sj+2].toarray()

            # 2x2 correlator block <c_i^\dagger c_j>
            C_ij = C[si:si+2, sj:sj+2]

            # Bond current (in units where e/ħ = 1)
            I_bond = -2.0 * np.imag(np.trace(T_ij @ C_ij))
            I_tot += I_bond

        current_list.append(I_tot)
        if(phi==0):
            print(f"current amplitude is{I_tot} ")
    current_list_operator = np.array(current_list)
    energy_list=np.array(energy_list)
    #calculate current using finite difference
    dphi = phi_list[1] - phi_list[0]
    current_list_dE_dphi = -(np.roll(energy_list, -1) - np.roll(energy_list, 1)) / (2 * dphi)
    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(phi_list, current_list_operator, marker='o',label='bond-current operator')
    plt.plot(phi_list, current_list_dE_dphi, linestyle='--', color='red',label='-dE/dphi')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlabel(r"Phase difference $\varphi$")
    plt.ylabel(r"Current $I(\varphi)$ (arb. units)")
    plt.title(rf"Josephson Current VS Phase Difference")
    # Save to project's plots/ directory with a safe filename (avoid $ and LaTeX in filenames)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"josephson_current_in_2_ways_alpha_{alpha}.jpg"
    save_path = PLOTS_DIR / safe_name
    plt.savefig(save_path, dpi=300)
    plt.tight_layout()
    print(f"Saved Josephson current plot to: {save_path}")
    #plt.show()
    return phi_list, current_list
def josephson_current_vs_vortice_number(Lx, Ly, t_hop, mass, mu, Delta0, L1, L2,
                                  x_cut=None, eps_pos=1e-10):
    W=L2-L1+1
    amp_list=[]
    for n in range(1,6):
        alpha = n / (W * Ly)
        phi_list,current_list=run_josephson_current_operator(Lx, Ly, t_hop,alpha, mass, mu, Delta0, L1, L2,)
        amp=np.max(np.array(current_list))
        amp_list.append(amp)
    plt.figure(figsize=(7, 5))
    plt.plot(range(1,6),amp_list,marker='o')
    plt.grid(alpha=0.3)
    plt.xlabel(r"Flux normalization $n$")
    plt.ylabel(r"Max. Josephson Current $I_max$")
    plt.title(rf"Josephson Current VS Flux Normalization")
    VORTICES_Y_GAUGE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VORTICES_Y_GAUGE_DIR / "josephson_current_vs_flux_normalization.jpg", dpi=300)
    _show_or_close()








def plot_bcs_dispersion(t=1.0, m=1.0, mu=0.3, Delta0=0.2,
                                     ky=0.0, num_k=401):
    """
    Plots BdG dispersion for the  1d Wilson–Dirac model along kx with fixed ky.

    Normal-state (2x2):
      h(k) = -t (σx sin kx + σy sin ky) + m σz (2 - cos kx - cos ky) - μ I

    Pairing:
      Δ = Δ0 (i σy)

    BdG (4x4):
      H_BdG(k) = [[ h(k),  Δ ],
                  [ Δ†,  -h(-k)^T ]]
    """

    # Pauli matrices (numpy)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def h_of_k(kx, ky):
        dx = -t * np.sin(kx)#x hopping term
        dy = -t * np.sin(ky)#y hopping term
        dz =  m * (2.0 - np.cos(kx) - np.cos(ky))#willson dirac mass term
        return dx * sx + dy * sy + dz * sz - mu * I2

    # spin- pairing
    Delta = Delta0 * (1j * sy)          # (0,-1;1,0)
    Delta_dag = Delta.conj().T          # transpose conjugate

    kx_list = np.linspace(-np.pi, np.pi, num_k)
    bands = np.zeros((num_k, 2), dtype=float)

    for idx, kx in enumerate(kx_list):
        hk = h_of_k(kx, ky)
        hmkT = h_of_k(-kx, -ky).T  # h(-k)^T

        Hbdg = np.block([
            [hk,        Delta],
            [Delta_dag, -hmkT]
        ])  # 4x4

        evals = np.linalg.eigvalsh(Hbdg)  # sorted
        negE, posE = evals[1], evals[2]
        bands[idx, 0] = negE.real  # hole band
        bands[idx, 1] = posE.real  # electron band
    # x-axis like your figure: k/(π/2) gives [-2,2] when k ∈ [-π,π]
    else:
        x = kx_list/np.pi
        xlabel = r"$k_x/pi$"

    plt.figure(figsize=(8, 5))
    for b in range(2):#plot all bands
        plt.plot(x, bands[:, b], linewidth=1.5)

    plt.axhline(0, linewidth=0.8)


    plt.grid(alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("Energy E")
    plt.plot(x,Delta0*np.ones_like(x),label=r'$\Delta$ ')
    plt.plot(x,-Delta0*np.ones_like(x),label=r'-$\Delta$ ')
    plt.title(f"BdG dispersion (ky={ky:.2f}) | t={t}, m={m}, μ={mu}, Δ={Delta0}")
    plt.tight_layout()
    plt.legend()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / f"bdg_dispersion_ky{ky:.2f}.jpg", dpi=300)
    _show_or_close()


import matplotlib.pyplot as plt
import numpy as np


def plot_lowest_positive_bdg_state():
    """
    Plots the lowest positive BdG (Bogoliubov-de Gennes) state densities for varying
    fluxes and exports the plots as an image.

    The function constructs a lattice simulation with specified boundary conditions and
    generates the Josephson junction Hamiltonian for different normalized fluxes n. It
    extracts the lowest positive energy eigenstate for each flux, computes the density
    maps, and visualizes these densities on a 2x3 grid layout plot. The densities
    correspond to the lowest positive energy state for each normalized flux n. The output
    includes both the data for these state densities and the generated visualization.

    Returns:
        List[Tuple[int, float, np.ndarray]]: A list of tuples for each normalized flux n.
            Each tuple contains:
            - n (int): The flux normalization factor.
            - E0 (float): The lowest positive energy eigenvalue for the flux.
            - rho_map (np.ndarray): A 2D array representing the density map of the lowest
              positive energy state, shaped according to the lattice dimensions.

    Raises:
        Warning: Prints a warning if no positive eigenvalues are found for a given flux n.
    """
    eps = 1e-10
    sim = LatticeSimulation(30, 30, boundary_condition_x='obc', boundary_condition_y='pbc')

    results = []
    # Updated to include n=6 (range is exclusive at the end, so 1 to 7 gives 1..6)
    n_values = range(1, 7)

    for n in n_values:
        L1 = 10
        L2 = 20
        W = L2 - L1 + 1

        H_bdg = sim.get_bdg_josephson_hamiltonian(
            t=1, m=0.3, mu=0.2, Delta0=0.2,
            phi_0=0, alpha=n / (W * sim.Ly), L1=L1, L2=L2
        )

        evals, evecs = sim.get_spectrum(H_bdg)

        pos_idx = np.where(evals > eps)[0]
        if len(pos_idx) == 0:
            print(f"Warning: No positive evals for n={n}")
            continue

        idx = pos_idx[np.argmin(evals[pos_idx])]
        E0 = float(evals[idx].real)

        psi = evecs[:, idx]
        dim_e = 2 * sim.N
        u = psi[:dim_e]
        u0 = u[0::2]
        u1 = u[1::2]

        rho = (np.square(np.abs(u0)) + np.square(np.abs(u1))).real
        if hasattr(rho, "A1"):
            rho = rho.A1
        else:
            rho = rho.flatten()

        rho_map = rho.reshape(sim.Ly, sim.Lx)
        results.append((n, E0, rho_map))

    # --- PLOTTING ---

    # Create a 2x3 grid (exactly 6 slots for 6 plots)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axs_flat = axs.flatten()

    for i, (n, E0, rho_map) in enumerate(results):
        ax = axs_flat[i]
        im = ax.imshow(rho_map, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(fr"$n={n}, E \approx {E0:.3e}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Lowest Positive Energy State Density for Different Fluxes")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "all_rho_maps_1to6.png", dpi=300)
    _show_or_close()

    return results

def plot_lowest_positive_bdg_phase():
    eps = 1e-10
    sim = LatticeSimulation(30, 30, boundary_condition_x='obc', boundary_condition_y='pbc')

    results = []
    # Loop from n=1 to n=6
    n_values = range(1, 7)

    for n in n_values:
        L1 = 10
        L2 = 20
        W = L2 - L1 + 1

        H_bdg = sim.get_bdg_josephson_hamiltonian(
            t=1, m=0.3, mu=0.2, Delta0=0.2,
            phi_0=0, alpha=n / (W * sim.Ly), L1=L1, L2=L2
        )

        evals, evecs = sim.get_spectrum(H_bdg)

        # Find lowest positive energy state
        pos_idx = np.where(evals > eps)[0]
        if len(pos_idx) == 0:
            print(f"Warning: No positive evals for n={n}")
            continue

        idx = pos_idx[np.argmin(evals[pos_idx])]
        E0 = float(evals[idx].real)

        # Extract eigenvector
        psi = evecs[:, idx]
        dim_e = 2 * sim.N
        u = psi[:dim_e]

        # Extract just the first spin component (u0)
        u0 = u[0::2]

        # Calculate Phase of u0
        # Reshape to 2D grid first
        u0_2d = u0.reshape(sim.Ly, sim.Lx)
        phase_map = np.angle(u0_2d)

        results.append((n, E0, phase_map))

    # --- PLOTTING ---

    # Create a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    axs_flat = axs.flatten()

    for i, (n, E0, phase_map) in enumerate(results):
        ax = axs_flat[i]
        # Use 'hsv' colormap for phase (cyclic)
        im = ax.imshow(phase_map, origin="lower", aspect="auto", cmap='hsv', vmin=-np.pi, vmax=np.pi)

        ax.set_title(fr"$n={n}, E \approx {E0:.3e}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Add colorbar for this specific subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Phase (rad)")

    plt.suptitle(r"Phase for Different Fluxes")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "all_phase_maps_1to6.png", dpi=300)
    _show_or_close()

    return results

def plot_spectrum_at_integers_only():
    # 1. Setup Simulation
    # We can afford a slightly larger lattice since we only run 7 steps
    sim = LatticeSimulation(30, 30, boundary_condition_x='obc', boundary_condition_y='pbc')
    L1, L2 = 10, 20
    W = L2 - L1

    # 2. Define Flux Grid (Integers ONLY)
    n_values = np.arange(0, 7)  # [0, 1, 2, 3, 4, 5, 6]

    # Storage
    all_fluxes = []
    all_energies = []
    minigaps = []

    print(f"Calculating spectrum for integer fluxes: {n_values}...")

    for n in n_values:
        alpha = n / (W * sim.Ly)

        # Build Hamiltonian (Using "Majorana Mode" parameters: m=0.6, mu=0.0)
        H_bdg = sim.get_bdg_josephson_hamiltonian(
            t=1, m=0.6, mu=0.0, Delta0=0.2,
            phi_0=0, alpha=alpha, L1=L1, L2=L2
        )

        # Solve Eigenvalues
        evals = np.linalg.eigvalsh(H_bdg.todense())

        # Filter for the "Gap Window"
        window = 0.4
        mask = (evals >= -window) & (evals <= window)
        gap_evals = evals[mask]

        all_energies.append(gap_evals)
        all_fluxes.append(np.full_like(gap_evals, n))

        # Minigap Calculation
        pos_evals = evals[evals > 1e-6]
        if len(pos_evals) > 0:
            minigaps.append(np.min(pos_evals))
        else:
            minigaps.append(0.0)

    # Flatten for plotting
    flat_fluxes = np.concatenate(all_fluxes)
    flat_energies = np.concatenate(all_energies)

    # --- FIGURE 1: Discrete Spectrum (The "Columns") ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Plot all states as black dots
    ax1.scatter(flat_fluxes, flat_energies, s=20, c='black', alpha=0.7, label='Andreev States')

    # Highlight the Zero Modes (states very close to E=0)
    # We look for states with |E| < 0.005
    zero_mode_mask = np.abs(flat_energies) < 0.005
    if np.any(zero_mode_mask):
        ax1.scatter(flat_fluxes[zero_mode_mask], flat_energies[zero_mode_mask],
                    s=50, c='red', zorder=10, label='Zero Modes (Majoranas)')

    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel("Energy $E$")
    ax1.set_xlabel(r"Flux Quanta $n$")
    ax1.set_title("Spectrum at Integer Fluxes around Zero Energy")
    ax1.set_xticks(n_values)  # Ensure x-axis only shows integers
    ax1.set_ylim(-0.3, 0.3)
    ax1.grid(True, alpha=0.3, axis='y')  # Grid lines for energy levels
    ax1.legend()

    plt.figure(fig1.number)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "fig1_discrete_spectrum.png", dpi=300)

    # --- FIGURE 2: Minigap at Integers ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    # We use a bar chart or scatter since the data is discrete
    ax2.plot(n_values, minigaps, color='crimson', linestyle='--', alpha=0.5)  # faint connecting line
    ax2.scatter(n_values, minigaps, color='crimson', s=50, zorder=10)

    # Highlight points that hit zero
    zero_gaps = np.array(minigaps) < 0.005
    if np.any(zero_gaps):
        ax2.scatter(n_values[zero_gaps], np.array(minigaps)[zero_gaps],
                    color='blue', s=80, marker='*', label='Topological Phase')

    ax2.set_ylabel(r"$E_{min}$")
    ax2.set_xlabel(r"Flux Quanta $n$")
    ax2.set_title("Lowest Excitation Energy")
    ax2.set_xticks(n_values)
    ax2.set_ylim(0, 0.15)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.figure(fig2.number)
    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "fig2_discrete_minigap.png", dpi=300)
    _show_or_close()


# Run it
#plot_spectrum_at_integers_only()

def plot_current_streamlines_gauge_invariant(n=4):
    print(f"--- Running Simulation for n={n} ---")
    # 1. Setup Simulation
    eps = 1e-10
    sim = LatticeSimulation(30, 30, boundary_condition_x='obc', boundary_condition_y='pbc')
    L1, L2 = 10,20
    W = L2 - L1
    # With current BdG phase profile phi_y = phi_0 + 4π * alpha * W * y,
    # using 1/(2*W*Ly) makes total winding over y equal 2π*n (not 4π*n).
    alpha = n / ( 2*W * sim.Ly)
    # 2. Build BdG Hamiltonian
    m_model = 0.3  # Choose a mass that gives a clear gap but still allows vortex modes (not too large)ע
    H_bdg = sim.get_bdg_josephson_hamiltonian(
        t=1, m=m_model, mu=0.05, Delta0=0.05,
        phi_0=np.pi/4, alpha=alpha, L1=L1, L2=L2
    )
    # This call will now use the FIXED get_spectrum from the class above
    evals, evecs = sim.get_spectrum(H_bdg)

    # 3. SMART SELECTION: Filter for Vortex Modes (Inside Junction)
    pos_indices = np.where(evals > eps)[0]
    sorted_pos_indices = pos_indices[np.argsort(evals[pos_indices])]

    selected_idx = None

    print(f"  Scanning low-energy states for Vortex modes...")

    for idx in sorted_pos_indices:
        psi = evecs[:, idx]
        psi_flat = np.asarray(psi).ravel()
        dim_e = 2 * sim.N
        u = psi_flat[:dim_e]
        v = psi_flat[dim_e:]
        # BdG basis is [u(2N), v(2N)], so build site density explicitly.
        density_on_site = (
            np.square(np.abs(u[0::2])) + np.square(np.abs(u[1::2])) +
            np.square(np.abs(v[0::2])) + np.square(np.abs(v[1::2]))
        ).real

        # Junction Mask
        x_coords = np.arange(sim.N) % sim.Lx
        in_junction = (x_coords >= L1) & (x_coords <= L2)

        # Ratio
        total_weight = np.sum(density_on_site)
        junction_weight = np.sum(density_on_site[in_junction])
        ratio = junction_weight / total_weight

        # If > 30% is in junction, we take it.
        if ratio > 0.5:
            selected_idx = idx
            print(f"    -> MATCH: Found Vortex Mode at E={evals[idx]:.5e} (Ratio={ratio:.2f})")
            break

    if selected_idx is None:
        print("  Warning: No vortex states found. Using lowest energy state.")
        selected_idx = sorted_pos_indices[0]

    # 4. Extract Spinors
    psi = np.asarray(evecs[:, selected_idx]).flatten()
    dim_e = 2 * sim.N
    u = psi[:dim_e]

    H0 = sim.get_dirac_hamiltonian(t=1, m=m_model, alpha=alpha, L1=L1, L2=L2).tocsr()

    Jx_bond = np.zeros((sim.Ly, sim.Lx - 1), dtype=float)
    Jy_bond = np.zeros((sim.Ly - 1, sim.Lx), dtype=float)

    def spinor_at(site):
        s = 2 * site
        return u[s:s + 2]

    # Calculate Currents (Explicit Math)
    for y in range(sim.Ly):
        for x in range(sim.Lx - 1):
            i_site = x + y * sim.Lx
            j_site = (x + 1) + y * sim.Lx
            si, sj = 2 * i_site, 2 * j_site

            T_ij = H0[si:si + 2, sj:sj + 2].toarray()
            ui = spinor_at(i_site)
            uj = spinor_at(j_site)

            # Use explicit conjugation instead of vdot to be safe
            val = np.dot(ui.conj(), np.dot(T_ij, uj))
            Jx_bond[y, x] = 2.0 * np.imag(val)

    for y in range(sim.Ly - 1):
        for x in range(sim.Lx):
            i_site = x + y * sim.Lx
            j_site = x + (y + 1) * sim.Lx
            si, sj = 2 * i_site, 2 * j_site

            T_ij = H0[si:si + 2, sj:sj + 2].toarray()
            ui = spinor_at(i_site)
            uj = spinor_at(j_site)

            val = np.dot(ui.conj(), np.dot(T_ij, uj))
            Jy_bond[y, x] = 2.0 * np.imag(val)

    # Average to sites
    Jx = np.zeros((sim.Ly, sim.Lx), dtype=float)
    Jy = np.zeros((sim.Ly, sim.Lx), dtype=float)

    Jx[:, 1:-1] = 0.5 * (Jx_bond[:, 1:] + Jx_bond[:, :-1])
    Jx[:, 0] = Jx_bond[:, 0];
    Jx[:, -1] = Jx_bond[:, -1]

    Jy[1:-1, :] = 0.5 * (Jy_bond[1:, :] + Jy_bond[:-1, :])
    Jy[0, :] = Jy_bond[0, :];
    Jy[-1, :] = Jy_bond[-1, :]

    Jmag = np.sqrt(Jx ** 2 + Jy ** 2)

    # 5. Plotting
    xgrid = np.arange(sim.Lx)
    ygrid = np.arange(sim.Ly)

    VORTICES_Y_GAUGE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    st = plt.streamplot(xgrid, ygrid, Jx, Jy, color=Jmag, density=2.0, arrowsize=1.5, cmap='viridis')
    plt.colorbar(st.lines, label="|J|")
    plt.title(f"Gauge-invariant Streamlines n={n}, alpha={alpha:.6e}\nE={evals[selected_idx]:.5e}")
    plt.xlabel("x");
    plt.ylabel("y")


    plt.savefig(VORTICES_Y_GAUGE_DIR / f"gauge_invariant_streamlines_n{n}.png", dpi=300)
    _show_or_close()
    print(f"  Completed Streamline simulation for n={n},saved to {VORTICES_Y_GAUGE_DIR / f'gauge_invariant_streamlines_n{n}.png'}")


# Run it

def plot_10_lowest_energy_eigenstates(Lx=30, Ly=30, n=4, t=1.0, m=0.3, mu=0.2, Delta0=0.2):
        sim = LatticeSimulation(Lx, Ly, boundary_condition_x='obc', boundary_condition_y='pbc')
        L1, L2 = 10, 20
        W = L2 - L1 + 1
        alpha = n / (W * sim.Ly)

        H_bdg = sim.get_bdg_josephson_hamiltonian(
            t=t, m=m, mu=mu, Delta0=Delta0,
            phi_0=0.0, alpha=alpha, L1=L1, L2=L2
        )

        # Use sparse near zero (much faster than dense)
        evals, evecs = sim.get_spectrum(H_bdg)

        # pick 10 smallest |E|
        idxs = np.argsort(np.abs(evals))[:10]

        results = []
        dim_e = 2 * sim.N

        for j, idx in enumerate(idxs, start=1):
            E = float(evals[idx].real)
            psi = evecs[:, idx]#eigen states
            u = psi[:dim_e]#top half of psi is electron spinors
            u0 = u[0::2]#first component
            u1 = u[1::2]#second component
            u0 = np.asarray(u0).ravel()
            u1 = np.asarray(u1).ravel()
            rho = (np.square(np.abs(u0)) + np.square(np.abs(u1))).real
            rho_map = rho.reshape(sim.Ly, sim.Lx)
            results.append((E, rho_map))
            print(f"State {j}: E = {E:.6e}")

        fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        axs = axs.flatten()

        for i, (E, rho_map) in enumerate(results):
            im = axs[i].imshow(rho_map, origin="lower", aspect="auto", cmap="viridis")
            axs[i].set_title(f"{i + 1}: E={E:.3e}", fontsize=10)
            plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

        VORTICES_Y_GAUGE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(VORTICES_Y_GAUGE_DIR / f"10_lowest_eigenstates_n{n}.png", dpi=300, bbox_inches="tight")
        _show_or_close()
        return results


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def bdg_site_density(sim, psi, use_u_plus_v=True):
    """
    psi: BdG eigenvector of length 4N in your basis [u(2N), v(2N)]
    Returns rho of length N.
    """
    psi = np.asarray(psi).ravel()
    N = sim.N
    dim_e = 2 * N

    u = psi[:dim_e]
    u0 = u[0::2]
    u1 = u[1::2]

    if use_u_plus_v:
        v = psi[dim_e:]
        v0 = v[0::2]
        v1 = v[1::2]
        rho = (np.square(np.abs(u0)) + np.square(np.abs(u1)) +
               np.square(np.abs(v0)) + np.square(np.abs(v1))).real
    else:
        rho = (np.square(np.abs(u0)) + np.square(np.abs(u1))).real

    return rho  # shape (N,)


def plot_top_states_by_R(
    Lx=30, Ly=30, n=5,
    t=1.0, m=0.3, mu=0.2, Delta0=0.2,
    L1=10, L2=20, phi_0=0.0,
    bc_x="obc", bc_y="pbc",
    K=10,
    Emin=0.0, Emax=0.15,   # energy window on positive energies
    use_u_plus_v=True,
    exclude_interfaces=True,
    out_dir=VORTICES_Y_GAUGE_DIR
):
    """
    Diagonalize BdG, compute R for each +E eigenstate, sort by R, and plot top K densities.

    exclude_interfaces=True uses x in [L1+1, L2-1] to avoid picking interface states.
    """
    sim = LatticeSimulation(Lx, Ly, boundary_condition_x=bc_x, boundary_condition_y=bc_y)

    # Use your current alpha convention (as in your streamline function)
    # NOTE: if you want strict y-PBC compatibility with phi_y = phi0 + 4π α (L2-L1) y,
    # a safer choice is alpha = n / (2*(L2-L1)*Ly). Keep your choice here for continuity:
    alpha = n / ((L2 - L1) * Ly)

    H_bdg = sim.get_bdg_josephson_hamiltonian(
        t=t, m=m, mu=mu, Delta0=Delta0,
        phi_0=phi_0, alpha=alpha, L1=L1, L2=L2
    )

    evals, evecs = sim.get_spectrum(H_bdg)  # make sure get_spectrum uses .toarray(), not .todense()

    # Build junction mask (sites)
    x_coords = np.arange(sim.N) % sim.Lx
    if exclude_interfaces:
        x_lo = L1 + 1
        x_hi = L2 - 1
        if x_hi < x_lo:  # very narrow junction: fallback
            x_lo, x_hi = L1, L2
    else:
        x_lo, x_hi = L1, L2

    in_junction = (x_coords >= x_lo) & (x_coords <= x_hi)

    # Compute R for positive-energy states in window
    records = []
    for idx, E in enumerate(evals):
        E = float(np.real(E))
        if E <= 0:
            continue
        if not (Emin <= E <= Emax):
            continue

        rho = bdg_site_density(sim, evecs[:, idx], use_u_plus_v=use_u_plus_v)
        R = float(rho[in_junction].sum() / rho.sum())
        records.append((R, E, idx))

    if not records:
        print("No states found in the requested energy window.")
        return []

    # Sort by R descending
    records.sort(key=lambda x: x[0], reverse=True)

    top = records[:K]
    print("Top states by junction weight R:")
    for rank, (R, E, idx) in enumerate(top, 1):
        print(f"{rank:2d}) idx={idx:4d}, E={E:.6e}, R={R:.3f}")

    # Plot
    ncols = 5 if K >= 5 else K
    nrows = int(np.ceil(K / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.6*nrows), constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()

    for i, (R, E, idx) in enumerate(top):
        rho = bdg_site_density(sim, evecs[:, idx], use_u_plus_v=use_u_plus_v)
        rho_map = rho.reshape(sim.Ly, sim.Lx)

        ax = axs[i]
        im = ax.imshow(rho_map, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"rank {i+1}: E={E:.2e}\nR={R:.2f}, idx={idx}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # optional: draw junction boundaries
        ax.axvline(L1-0.5, color="w", lw=1, alpha=0.6)
        ax.axvline(L2+0.5, color="w", lw=1, alpha=0.6)

    # Turn off unused axes
    for j in range(len(top), len(axs)):
        axs[j].axis("off")

    # Save
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    fname = f"top_by_R_n{n}_Ewin_{Emin:.3f}_{Emax:.3f}_K{K}.png"
    plt.savefig(outp / fname, dpi=300, bbox_inches="tight")
    _show_or_close()

    return top


if __name__ == "__main__":
    #top = plot_top_states_by_R(Lx=30, Ly=30, n=5, K=10, Emin=0.0, Emax=0.12)

    #results = plot_lowest_positive_bdg_state()
    #plot_lowest_positive_bdg_phase()

    #plot_spectrum_at_integers_only()

    # = plot_10_lowest_energy_eigenstates(Lx=30, Ly=30, n=5)

    #plot_bcs_dispersion()
    #run_bcs_simulation(40,40,1.0,2,1,1,'pbc')
    #run_simulation()
    #run_josephson_simulation(30,30,1.0,2,0.3,0.2,10,20,'pbc')
    run_josephson_current_operator(30,30,1.0,0,0.4,0.1,0.1,10,20)
    #josephson_current_vs_vortice_number(30,30,1.0,0.3,0.2,0.2,10,20)
   # plot_current_streamlines_gauge_invariant(n=6 )
