def __init__(self, Lx, Ly, lattice_constant=1.0, boundary_condition='pbc'):
    """
    Initialize the lattice parameters.

    Args:
        Lx (int): Number of sites in x-direction
        Ly (int): Number of sites in y-direction
        lattice_constant (float): Physical distance between  lattice sites default 1
        boundary_condition (str): 'pbc' for Periodic, 'obc' for Open
    """
    self.Lx = Lx
    self.Ly = Ly
    self.a = lattice_constant
    self.bc = boundary_condition.lower()
    self.N = Lx * Ly  # Total number of unit cells

    # Build neighbor matrices (Hopping connections)
    # We separate X and Y hoppings because Dirac models depend on direction
    self.hop_x, self.hop_y = self._build_hopping_matrices()

    # Total Adjacency is the sum of directional hoppings (+ their hermitian conjugate)
    self.adjacency = (self.hop_x + self.hop_y) + (self.hop_x + self.hop_y).T.conj()


def _build_hopping_matrices(self):
    """
    Constructs sparse hopping matrices for X and Y directions
    handling both PBC and OBC.
    """
    # Create empty lists to construct sparse matrix (COO format)
    data_x, row_x, col_x = [], [], []
    data_y, row_y, col_y = [], [], []

    for y in range(self.Ly):
        for x in range(self.Lx):
            # Current site index (flattened)
            i = x + y * self.Lx

            # --- Neighbor in +x direction ---
            x_next = x + 1
            add_x = False

            if x_next < self.Lx:
                # check if not edge
                add_x = True
            elif self.bc == 'pbc':
                # Boundary connection (wrap around)
                x_next = 0
                add_x = True
            # If OBC and at edge, add_x remains False

            if add_x:
                j_x = x_next + y * self.Lx
                row_x.append(i)
                col_x.append(j_x)
                data_x.append(1.0)

            # --- Neighbor in +y direction ---
            y_next = y + 1
            add_y = False

            if y_next < self.Ly:
                # Bulk connection
                add_y = True
            elif self.bc == 'pbc':
                # Boundary connection (wrap around)
                y_next = 0
                add_y = True

            if add_y:
                j_y = x + y_next * self.Lx
                row_y.append(i)
                col_y.append(j_y)
                data_y.append(1.0)

    # Create sparse matrices
    H_x = sparse.coo_matrix((data_x, (row_x, col_x)), shape=(self.N, self.N))
    H_y = sparse.coo_matrix((data_y, (row_y, col_y)), shape=(self.N, self.N))

    return H_x, H_y