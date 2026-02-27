import numpy as np
import scipy.sparse as sparse
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

    def _peierls_hop_y_in_junction(self, alpha, L1, L2):
        """
        Computes the Peierls phase factor for hopping terms in the y-direction within
        a specified junction region, with an applied magnetic flux characterized
        by the parameter alpha. The resulting modified hopping matrix considers
        phase shifts only in the specified junction range along the x-direction.

        Parameters:
        alpha: float
            The magnetic flux parameter. Determines the phase factor applied to the
            hopping terms. A value of 0 means no magnetic field is applied.

        L1: int
            The starting x-coordinate of the junction region where the Peierls
            phase modification is applied.

        L2: int
            The ending x-coordinate of the junction region where the Peierls
            phase modification is applied. The phase is applied for bonds starting
            in the range [L1, L2-1].

        Returns:
        scipy.sparse.csr_matrix
            A sparse matrix representing the modified hopping terms in the
            y-direction with the applied Peierls phase factors.
        """
        if abs(alpha) < 1e-15:
            return self.hop_y  # B=0 => original

        hy = self.hop_y.tocoo()
        data = hy.data.astype(complex).copy()
        W=L2-L1+1
        x_mid=(L1+L2)/2.0  # center of junction for symmetric phase profile
        for n in range(len(data)):
            i = int(hy.row[n])  # from-site index
            x = i % self.Lx
            y = i // self.Lx
            if x < L1:
                theta = (L1 - x_mid)  # constant left positive phase shift
            elif x <= L2:
                theta = (x - x_mid)  # linear ramp inside junction
            else:
                theta = (L2 - x_mid)  # constant right negative phase shift
            data[n] *= np.exp(2j * np.pi * alpha * theta)
        return sparse.coo_matrix((data, (hy.row, hy.col)), shape=hy.shape).tocsr()

    def get_bdg_josephson_hamiltonian(self, t=1.0, m=0.0, mu=0.0,
                                      Delta0=0.2, phi_0=0.0,
                                      alpha=0.0, L1=10, L2=20):
        """
        Generates the Bogoliubov-de Gennes (BdG) Hamiltonian for a Josephson junction.

        This method constructs the BdG Hamiltonian for a Dirac material Josephson
        junction with specified parameters. The Hamiltonian includes the normal state
        Dirac Hamiltonian, the chemical potential, and a superconducting pairing term
        with a phase difference across the junction.

        The method assumes periodic boundary conditions in the y-direction for consistent
        quantization. The function also verifies the compatibility of the system's
        geometrical and physical parameters with the specified setup, raising exceptions
        if necessary.

        Parameters:
            t (float): Hopping amplitude in the x and y directions.
            m (float): On-site mass term in the normal state Hamiltonian.
            mu (float): Chemical potential term adjusting the Fermi energy.
            Delta0 (float): Magnitude of the superconducting gap.
            phi_0 (float): Initial phase difference of the superconducting order parameter.
            alpha (float): Phase gradient per y-layer in units of 2π.
            L1 (int): Left boundary of the Josephson junction region in x-sites.
            L2 (int): Right boundary of the Josephson junction region in x-sites.

        Returns:
            scipy.sparse.csr_matrix: The resulting BdG Hamiltonian in compressed sparse row format.
            scipy.sparse.csr_matrix: The resulting BdG Hamiltonian in compressed sparse row format.

        Raises:
            ValueError: If the specified lengths L1 and L2 are invalid for the given system
                        dimensions (e.g., L1 and L2 out of bounds, L1 >= L2, etc.).
        """
        # Normal state with Peierls ONLY in junction y-hops (A = (0,Bx,0))
        H0 = self.get_dirac_hamiltonian(t=t, m=m, alpha=alpha, L1=L1, L2=L2).tocsr()
        dim = H0.shape[0]  # 2N
        I2N = sparse.eye(dim, format="csr", dtype=complex)
        H0_mu = H0 - mu * I2N

        sigma_y = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)

        if (L1 >= self.Lx) or (L2 >= self.Lx) or (L1 < 0) or (L2 < 0) or (L2 <= L1):
            raise ValueError("Invalid Josephson junction lengths L1 and L2.")

        W = (L2 - L1 ) # junction width in x sites (matches your Δ=0 for L1<=x<=L2)
        # (Optional but recommended) enforce PBC-in-y quantization:
        # alpha * W * Ly must be integer for Δ(y) to be periodic.
        print (self.Ly)
        print (W)
        q = alpha * W * self.Ly
        if abs(q - round(q)) > 1e-8 and abs(alpha) > 1e-15:
            print(f"Warning: PBC in y prefers n= alpha*W*Ly integer; got {q:.6f}")
        Delta_blocks = []
        for y in range(self.Ly):
            # y-dependent phase difference (left SC vs right SC)
            # FIX: use W consistently instead of (L2 - L1)
            phi_y = phi_0 +4.0 * np.pi * alpha * W * y  # phase difference grows linearly with y due to A=(0,Bx,0)
            for x in range(self.Lx):
                #phase profile :left SC: postive phase phi/2 , right SC: negative phase -phi/2
                if x < L1:
                    Delta_site = 1j * Delta0 * sigma_y*np.exp(-0.5j*phi_y)
                elif L1 <= x <= L2:
                    # junction: Δ=0
                    Delta_site = sparse.csr_matrix((2, 2), dtype=complex)
                else:
                    # right SC: negative  phase A
                    Delta_site = 1j * Delta0 * sigma_y*np.exp(0.5j*phi_y)
                Delta_blocks.append(Delta_site)

        Delta_block = sparse.block_diag(Delta_blocks, format="csr")  # 2N x 2N

        H_bdg = sparse.bmat(
            [[H0_mu, Delta_block],
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



    def get_spectrum(self, hamiltonian):
        """
        Diagonalize the Hamiltonian to find all energy eigenstates.
        Uses dense linear algebra for accuracy on small/medium lattices.
        returns: eigen_energies, eigen_states of the Hamiltonian
        """
        # Convert to dense for full diagonalization (scipy.linalg.eigh)
        H_dense = hamiltonian.toarray()
        eigen_energies, eigen_states = np.linalg.eigh(H_dense)
        return eigen_energies,eigen_states

    def get_dirac_hamiltonian(self, t=1.0, m=0.0, alpha=0.0, L1=None, L2=None):
        # Pauli Matrices
        sigma_x = sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
        sigma_y = sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)

        # --- choose hopping matrices ---
        if (L1 is not None) and (L2 is not None):
            hop_y_eff = self._peierls_hop_y_in_junction(alpha, L1, L2)  # phases only on x-links in junction
        else:
            hop_y_eff = self.hop_y
        hop_x_eff = self.hop_x  # IMPORTANT: unchanged in A=(0,Bx,0)

        # Your existing kinetic terms (kept as-is)
        T_x = (-1j * t) * sigma_x - (m / 2) * sigma_z
        H_kin_x = sparse.kron(hop_x_eff, T_x) + sparse.kron(hop_x_eff, T_x).getH()

        T_y = (-1j * t) * sigma_y - (m / 2) * sigma_z
        H_kin_y = sparse.kron(hop_y_eff, T_y) + sparse.kron(hop_y_eff, T_y).getH()

        H_mass = sparse.kron(sparse.eye(self.N, format="csr"), 2 * m * sigma_z)

        return (H_kin_x + H_kin_y + H_mass).tocsr()
