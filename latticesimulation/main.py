import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from pathlib import Path
import scipy.sparse.linalg as sla


class LatticeSimulation:
    # python
    def __init__(self, Lx, Ly, lattice_constant=1.0,
                 boundary_condition='pbc',
                 boundary_condition_x=None,
                 boundary_condition_y=None):
        """
        Initialize the lattice parameters.

        Args:
            Lx (int): Number of sites in x-direction
            Ly (int): Number of sites in y-direction
            lattice_constant (float): Physical distance between lattice sites (default 1.0)
            boundary_condition (str): fallback BC if per-axis is not provided ('pbc' or 'obc')
            boundary_condition_x (str|None): 'pbc' or 'obc' for x-axis
            boundary_condition_y (str|None): 'pbc' or 'obc' for y-axis
            B_amp (float): Amplitude of the magnetic field around the lattice

        """
        self.Lx = Lx
        self.Ly = Ly
        self.a = lattice_constant

        # Backward compatibility: if per-axis not provided, use `boundary_condition`
        bc_fallback = (boundary_condition or 'pbc').lower()
        self.bc_x = (boundary_condition_x or bc_fallback).lower()
        self.bc_y = (boundary_condition_y or bc_fallback).lower()

        self.N = Lx * Ly  # Total number of unit cells

        # Build neighbor matrices (Hopping connections)
        self.hop_x, self.hop_y = self._build_hopping_matrices()

        # Total Adjacency is the sum of directional hoppings (+ their hermitian conjugate)
        self.adjacency = (self.hop_x + self.hop_y) + (self.hop_x + self.hop_y).T.conj()

    def _build_hopping_matrices(self):
        """
        Constructs sparse hopping matrices for X and Y directions
        handling both per-axis PBC and OBC.
        """
        import scipy.sparse as sparse

        data_x, row_x, col_x = [], [], []
        data_y, row_y, col_y = [], [], []

        for y in range(self.Ly):
            for x in range(self.Lx):
                i = x + y * self.Lx

                # --- Neighbor in +x direction ---
                x_next = x + 1
                add_x = False

                if x_next < self.Lx:
                    add_x = True
                elif self.bc_x == 'pbc':
                    x_next = 0
                    add_x = True
                # If OBC in x and at edge, add_x remains False

                if add_x:
                    j_x = x_next + y * self.Lx
                    row_x.append(i)
                    col_x.append(j_x)
                    data_x.append(1.0)

                # --- Neighbor in +y direction ---
                y_next = y + 1
                add_y = False

                if y_next < self.Ly:
                    add_y = True
                elif self.bc_y == 'pbc':
                    y_next = 0
                    add_y = True
                # If OBC in y and at edge, add_y remains False

                if add_y:
                    j_y = x + y_next * self.Lx
                    row_y.append(i)
                    col_y.append(j_y)
                    data_y.append(1.0)

        H_x = sparse.coo_matrix((data_x, (row_x, col_x)), shape=(self.N, self.N))
        H_y = sparse.coo_matrix((data_y, (row_y, col_y)), shape=(self.N, self.N))
        return H_x, H_y



    def get_bdg_josephson_hamiltonian(self, t=1.0, m=0.0, mu=0.0, Delta0=0.2,phi_josephson=0.0,L1=10,L2=20,B_amp):
        """
        Build BdG Hamiltonian from Wilson–Dirac H0.
        Size: (4N x 4N) since H0 is (2N x 2N)
        Delta profile to model a Josephson junction along x-direction:
        Delta=Delta_0*exp(i*phi) for x<L1
        Delta=0 for L1<=x<=L2
        Delta=Delta_0 for x>L2
        B_amp- magnetic field amplitude
        """
        # Normal state (2N x 2N)
        H0 = self.get_dirac_hamiltonian(t=t, m=m).tocsr()

        dim = H0.shape[0]  # = 2N
        I2N = sparse.eye(dim, format="csr", dtype=complex)

        # Chemical potential shift
        H0_mu = H0 - mu * I2N

        # Pairing matrix Δ (2N x 2N): onsite Δ0 * (i σ_y) on each site
        sigma_y = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
        if(L1>=self.Lx or L2>self.Lx or L1<0 or L2<0 or L2<=L1):
            raise ValueError("Invalid Josephson junction lengths L1 and L2.")
        Delta_blocks = []
        for y in range(self.Ly):
            for x in range(self.Lx):
                if x < L1:
                    Delta_site = 1j * Delta0 * np.exp(1j * phi_josephson) * sigma_y
                elif L1 <= x <= L2:#between SC regions delta=0
                    Delta_site = sparse.csr_matrix([[0, 0], [0, 0]], dtype=complex)
                else:  # x > L2
                    Delta_site = 1j * Delta0 * sigma_y
                Delta_blocks.append(Delta_site)
        #Delta_block = 1j * Delta0 * sigma_y                     # 2x2
        Delta_block = sparse.block_diag(Delta_blocks, format="csr")  # 2N x 2N
        #Delta = sparse.kron(sparse.eye(self.N, format="csr"), Delta_block, format="csr")

        # BdG block matrix (4N x 4N)
        H_bdg = sparse.bmat(
            [[H0_mu,        Delta_block],
             [Delta_block.getH(), -H0_mu.transpose()]],
            format="csr"
        )
        return H_bdg

    def get_schrodinger_hamiltonian(self, t=1.0, epsilon=0.0):
        """
        Constructs a standard Tight-Binding Hamiltonian.
        H = -t * sum(<i,j>) (c_i^dag c_j) + epsilon * sum(c_i^dag c_i)
        #epsilon is the onsite interaction term.

        This results in a parabolic dispersion near the band bottom (E ~ k^2).
        """
        H_hop = -t * self.adjacency

        # Onsite potential
        H_onsite = epsilon * sparse.eye(self.N)

        return H_hop + H_onsite

    def get_dirac_hamiltonian(self, t=1.0, m=0.0):
        """
        Constructs a discretized Dirac Hamiltonian on a lattice as per the image.
        Requires 2 degrees of freedom (orbitals/spin) per site.

        Matches the formula: H(k) = -2*t * (sigma_x*sin(kx) + sigma_y*sin(ky))
        Here, t corresponds to 2*lambda.

        H = +i(t/2) * [ sigma_x * (Hop_Right - Hop_Left) ... ]
        """
        # Pauli Matrices
        sigma_x = sparse.csr_matrix([[0, 1], [1, 0]])
        sigma_y = sparse.csr_matrix([[0, -1j], [1j, 0]])
        sigma_z = sparse.csr_matrix([[1, 0], [0, -1]])

        # 1. Kinetic Term (Hopping)
        # We use a hopping term T such that H_hop = T * hop_mat + h.c.
        # To match the image's "i * sigma * lambda", we use +1j here.

        # X-Hopping: T_x = +i * (t/2) * sigma_x
        # When Fourier transformed, this gives -t * sigma_x * sin(k)
        T_x = -1j * t * sigma_x-m/2*sigma_z
        H_kin_x = sparse.kron(self.hop_x, T_x)
        H_kin_x = H_kin_x + H_kin_x.getH()  # Adds the reverse hopping (-i)

        # Y-Hopping: T_y = +i * (t/2) * sigma_y
        T_y = -1j * t* sigma_y-m/2*sigma_z
        H_kin_y = sparse.kron(self.hop_y, T_y)
        H_kin_y = H_kin_y + H_kin_y.getH()

        # 2. Mass Term (Onsite)
        # M = I_lattice (kron) (m * sigma_z)
        H_mass = sparse.kron(sparse.eye(self.N), 2*m * sigma_z)

        H_total = H_kin_x + H_kin_y + H_mass
        return H_total

    def get_spectrum(self, hamiltonian):
        """
        Diagonalize the Hamiltonian to find all energy eigenstates.
        Uses dense linear algebra for accuracy on small/medium lattices.
        returns: eigen_energies, eigen_states of the Hamiltonian
        """
        # Convert to dense for full diagonalization (scipy.linalg.eigh)
        H_dense = hamiltonian.todense()
        eigen_energies, eigen_states = np.linalg.eigh(H_dense)
        return eigen_energies,eigen_states

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
    out_dir = Path(__file__).resolve().parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(out_dir / f"dirac_real_vs_numerical.jpg", dpi=300)
    plt.show()

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
    plt.savefig("bdg_pbc.jpg", dpi=300)
    plt.show()
    print(f"  Completed BCS simulation for {bc.upper()}.")
    return energies_bdg

def run_josephson_current_operator(Lx, Ly, t_hop, mass, mu, Delta0, L1, L2, bc,
                                  x_cut=None, eps_pos=1e-10):
    """
    Compute Josephson current I(φ) in 2 ways: the bond-current operator expectation value, and
    using josephson relation I=-DeltaE/DeltaPhi.

    In the operator method We sum currents across a cut: (x_cut,y) -> (x_cut+1,y) for all y.
    Choose x_cut inside the weak link for a clean "junction current".
    """
    print(f"Running Josephson current-operator simulation for {bc.upper()}...")

    sim = LatticeSimulation(Lx=Lx, Ly=Ly, boundary_condition_x='obc', boundary_condition_y='pbc')

    # Choose a cut inside the junction by default (middle of [L1, L2])
    if x_cut is None:
        x_cut = (L1 + L2) // 2
        x_cut = max(0, min(Lx - 2, x_cut))  # ensure valid

    phi_list = np.linspace(0, 2*np.pi, 21, endpoint=True)
    current_list = []

    # Normal-state Hamiltonian (electron sector), used to extract hopping blocks T_ij
    H0 = sim.get_dirac_hamiltonian(t=t_hop, m=mass).tocsr()

    dim_e = 2 * sim.N  # electron-sector dimension (2 internal comps per site)
    energy_list=[]

    for phi in phi_list:
        # Build BdG for this phase
        H_bdg = sim.get_bdg_josephson_hamiltonian(
            t=t_hop, m=mass, mu=mu, Delta0=Delta0,
            phi_josephson=phi, L1=L1, L2=L2
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
            I_bond = 2.0 * np.imag(np.trace(T_ij @ C_ij))
            I_tot += I_bond

        current_list.append(I_tot)

    current_list_operator = np.array(current_list)
    energy_list=np.array(energy_list)
    #calculate current using finite difference
    dphi=phi_list[1]-phi_list[0]
    current_list_dE_dphi=-np.gradient(energy_list,dphi)
    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(phi_list, current_list_operator, marker='o',label='bond-current operator')
    plt.plot(phi_list, current_list_dE_dphi, linestyle='--', color='red',label='-dE/dphi')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlabel(r"Phase difference $\varphi$")
    plt.ylabel(r"Current $I(\varphi)$ (arb. units)")
    plt.title(f"Josephson Current VS Phase Difference")
    plt.savefig("josephson_current_in_2_ways.jpg", dpi=300)
    plt.tight_layout()
    plt.show()

    return phi_list, current_list







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
    plt.savefig(f"bdg_dispersion_ky{ky:.2f}.jpg", dpi=300)
    plt.show()



#plot_bcs_dispersion()
run_bcs_simulation(40,40,1.0,2,1,1,'pbc')
#run_simulation()
#run_josephson_simulation(30,30,1.0,2,0.3,0.2,10,20,'pbc')
#run_josephson_current_operator(30,30,1.0,2,0.3,0.2,10,20,'pbc')