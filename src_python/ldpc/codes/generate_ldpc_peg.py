import numpy as np
import scipy.sparse
from collections import deque

def generate_ldpc_peg(m: int, n: int, dv: int, dc: int) -> scipy.sparse.csr_matrix:
    """
    Generate an (m x n) LDPC parity-check matrix using the Progressive Edge-Growth (PEG) algorithm.

    Parameters:
        m (int): Number of check nodes (rows).
        n (int): Number of variable nodes (columns).
        dv (int): Degree of each variable node (number of edges per variable).
        dc (int): Maximum degree of each check node (capacity per check).

    Returns:
        scipy.sparse.csr_matrix: The generated LDPC parity-check matrix in CSR format.

    The PEG algorithm incrementally builds a Tanner graph by adding edges one at a time
    to maximize the girth (length of shortest cycle) of the bipartite graph. Each variable
    node connects to dv checks, choosing the "farthest" available check to avoid short cycles.
    """
    # Ensure total capacity is sufficient
    if n * dv > m * dc:
        raise ValueError(f"Insufficient capacity: n*dv ({n*dv}) > m*dc ({m*dc})")

    # Initialize the adjacency matrix and degree trackers
    H = np.zeros((m, n), dtype=np.int8)    # H[c, v] = 1 if check c connects to variable v
    deg_v = np.zeros(n, dtype=int)         # degree of each variable node
    deg_c = np.zeros(m, dtype=int)         # degree of each check node

    # Adjacency lists for BFS
    var_to_checks = [[] for _ in range(n)]  # checks connected to each variable
    check_to_vars = [[] for _ in range(m)]  # variables connected to each check

    def bfs_distances(start_v: int) -> np.ndarray:
        """
        Compute shortest-path distances from variable node start_v to all check nodes
        in the current partial graph using breadth-first search.
        Unconnected checks remain at distance -1.
        """
        dist_c = -np.ones(m, dtype=int)
        visited_vars = [False] * n
        queue = deque()

        # Mark start variable and enqueue its direct check neighbors
        visited_vars[start_v] = True
        for c in var_to_checks[start_v]:
            dist_c[c] = 1
            queue.append(('c', c))

        while queue:
            kind, idx = queue.popleft()
            if kind == 'c':
                # From a check node, go to connected variables
                for vv in check_to_vars[idx]:
                    if not visited_vars[vv]:
                        visited_vars[vv] = True
                        # Then from each variable, go to its checks
                        for cc in var_to_checks[vv]:
                            if dist_c[cc] == -1:
                                dist_c[cc] = dist_c[idx] + 2
                                queue.append(('c', cc))
        return dist_c

    # Main PEG loop: add dv edges per variable node
    for v in range(n):
        for _ in range(dv):
            # Compute distances to all checks
            dists = bfs_distances(v)
            INF = m + n  # treat unreachable as infinite
            effective = np.where(dists >= 0, dists, INF)

            # Choose check nodes at maximum distance
            max_dist = effective.max()
            candidates = [c for c, d in enumerate(effective) if d == max_dist]

            # Exclude already connected or full checks
            eligible = [c for c in candidates if deg_c[c] < dc and c not in var_to_checks[v]]
            # Fallback: allow any check with capacity
            if not eligible:
                eligible = [c for c in range(m) if deg_c[c] < dc]
            # Fallback: allow all checks to prevent emptiness
            if not eligible:
                eligible = list(range(m))

            # From eligible, pick the least-used
            best_c = min(eligible, key=lambda c: deg_c[c])

            # Connect v to best_c
            H[best_c, v] = 1
            deg_v[v] += 1
            deg_c[best_c] += 1
            var_to_checks[v].append(best_c)
            check_to_vars[best_c].append(v)

    # Return sparse parity-check matrix
    return scipy.sparse.csr_matrix(H)