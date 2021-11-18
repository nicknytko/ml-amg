import networkx as nx
import torch
import torch_geometric as tg
import numpy as np
import scipy.sparse as sp
import pyamg
import matplotlib.pyplot as plt
import matplotlib

import shapely.geometry as sg
from shapely.ops import cascaded_union

def graph_from_matrix(A, agg_op):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    # Create cluster feature
    clusters = np.array(agg_op.argmax(axis=1)).flatten()
    cluster_adj = {} # 0 if in same cluster, 1 if not
    for (u, v) in nx.edges(G):
        adj = (0 if (clusters[u] == clusters[v]) else 1)
        cluster_adj[(u, v)] = adj

    nx.set_edge_attributes(G, cluster_adj, 'cluster_adj')
    nx_data = tg.utils.from_networkx(G, None, ['weight', 'cluster_adj'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))

def graph_from_matrix_basic(A):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    # Create cluster feature
    cluster_adj = {} # 0 if in same cluster, 1 if not
    for (u, v) in nx.edges(G):
        cluster_adj[(u, v)] = 1.0 / n

    nx.set_edge_attributes(G, cluster_adj, 'cluster_adj')
    nx_data = tg.utils.from_networkx(G, None, ['weight', 'cluster_adj'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))

class Grid():
    def __init__(self, A_csr, x=None):
        '''
        Initializes the grid object

        Parameters
        ----------
        A_csr : scipy.sparse.csr_matrix
          CSR matrix representing the underlying PDE
        x : numpy.ndarray
          Positions of the points of each node.  Should have shape (n_pts, n_dim).
        '''

        self.A = A_csr
        self.x = x

    @property
    def networkx(self):
        return nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)

    def plot(self, ax=None):
        '''
        Plot the nodes and edges of the sparse matrix.

        Parameters
        ----------
        ax : axis
          matplotlib axis
        '''

        graph = self.networkx
        if self.x is None:
            positions = None
        else:
            positions = {}
            for node in graph.nodes:
                positions[node] = self.x[node]

        nx.drawing.nx_pylab.draw_networkx(graph, ax=ax, pos=positions, arrows=False, with_labels=False, node_size=100)

    def plot_agg(self, AggOp, ax=None, color=None, edgecolor='0.5', lw=1):
        '''
        Aggregate visualization borrowed/stolen from PyAMG
        (https://github.com/pyamg/pyamg/blob/main/Docs/logo/pyamg_logo.py)

        Parameters
        ----------
        AggOp : CSR sparse matrix
          n x nagg encoding of the aggregates AggOp[i,j] == 1 means node i is in aggregate j
        ax : axis
          matplotlib axis
        color : string
          color of the aggregates
        edgecolor : string
          color of the aggregate edges
        lw : float
          line width of the aggregate edges
        '''

        if ax is None:
            ax = plt.gca()

        for agg in AggOp.T:                                    # for each aggregate
            aggids = agg.indices                               # get the indices

            todraw = []                                        # collect things to draw
            if len(aggids) == 1:
                i = aggids[0]
                coords = (self.x[i, 0], self.x[i,1])
                newobj = sg.Point(coords)
                todraw.append(newobj)

            for i in aggids:                                   # for each point in the aggregate
                nbrs = self.A.getrow(i).indices                # get the neighbors in the graph

                for j1 in nbrs:                                # for each neighbor
                    found = False                              # mark if a triad ("triangle") is found
                    for j2 in nbrs:
                        if (j1!=j2 and i!=j1 and i!=j2 and     # don't count i - j - j as a triangle
                            j1 in aggids and j2 in aggids and  # j1/j2 are in the aggregate
                            self.A[j1,j2]                      # j1/j2 are connected
                            ):
                            found = True                       # i - j1 - j2 are in the aggregate
                            coords = list(zip(self.x[[i,j1,j2], 0], self.x[[i,j1,j2],1]))
                            todraw.append(sg.Polygon(coords))  # add the triangle to the list
                    if not found and i!=j1 and j1 in aggids:   # if we didn't find a triangle, then ...
                        coords = list(zip(self.x[[i,j1], 0], self.x[[i,j1],1]))
                        newobj = sg.LineString(coords)         # add a line object to the list
                        todraw.append(newobj)

            todraw = cascaded_union(todraw)                    # union all objects in the aggregate
            todraw = todraw.buffer(0.08)                        # expand to smooth
            todraw = todraw.buffer(-0.05)                      # then contract

            try:
                xs, ys = todraw.exterior.xy                    # get all of the exterior points
                ax.fill(xs, ys, clip_on=False, alpha=0.7)      # fill with a color
            except:
                pass                                           # don't plot singletons

    def structured_1d_poisson_dirichlet(n, xdim=(0,1)):
        '''
        Creates a 1D poisson system on a structured grid, discretized using finite differences.
        Dirichlet boundary conditions are assumed.

        Parameters
        ----------
        n : integer
          Number of interior points
        xdim : tuple (float, float)
          Left and right-most x values of the domain

        Returns
        -------
        Grid object with given parameters.
        '''

        x = np.linspace(xdim[0], xdim[1], n+2)[1:-1]
        h = abs(x[1] - x[0])
        A = (sp.eye(n)*2 - sp.eye(n,k=-1) - sp.eye(n,k=1))*(h**-2.)

        return Grid(A.tocsr(), np.column_stack((x, np.zeros_like(x))))

    def structured_1d_poisson_neumann(n, xdim=(0,1)):
        '''
        Creates a 1D poisson system on a structured grid, discretized using finite differences.
        Neumann boundary conditions are assumed.

        Parameters
        ----------
        n : integer
          Number of total points on domain
        xdim : tuple (float, float)
          Left and right-most x values of the domain

        Returns
        -------
        Grid object with given parameters.
        '''

        x = np.linspace(xdim[0], xdim[1], n)
        h = abs(x[1] - x[0])
        A = sp.eye(n)*2 - sp.eye(n,k=-1) - sp.eye(n,k=1)

        # Apply neumann conditions
        A = A.tolil()
        A[0,0] = 1; A[0,1] = -1
        A[-1,-1] = 1; A[-1,-2] = -1;

        # Apply scaling
        A = A.tocsr() * (h**-2.)

        return Grid(A, np.column_stack((x, np.zeros_like(x))))

    def structured_2d_poisson_dirichlet(n_pts_x, n_pts_y,
                                        xdim=(0,1), ydim=(0,1),
                                        epsilon=1.0, theta=0.0):
        '''
        Creates a 2D poisson system on a structured grid, discretized using finite elements.
        Dirichlet boundary conditions are assumed.

        Parameters
        ----------
        n_pts_x : integer
          Number of inner points in the x dimension (not including boundary points)
        n_pts_y : integer
          Number of inner points in the y dimension (not including boundary points)
        xdim : tuple (float, float)
          Bounds for domain in x dimension.  Represents smallest and largest x values.
        ydim : tuple (float, float)
          Bounds for domain in y dimension.  Represents smallest and largest y values.

        Returns
        -------
        Grid object with given parameters.
        '''

        x_pts = np.linspace(xdim[0], xdim[1], n_pts_x+2)[1:-1]
        y_pts = np.linspace(xdim[0], ydim[1], n_pts_y+2)[1:-1]
        delta_x = abs(x_pts[1] - x_pts[0])
        delta_y = abs(y_pts[1] - y_pts[0])

        xx, yy = np.meshgrid(x_pts, y_pts)
        xx = xx.flatten()
        yy = yy.flatten()

        grid_x = np.column_stack((xx, yy))
        n = n_pts_x * n_pts_y
        A = sp.lil_matrix((n, n), dtype=np.float64)

        stencil = pyamg.gallery.diffusion_stencil_2d(epsilon=epsilon, theta=theta, type='FD')
        print(stencil)

        for i in range(n_pts_x):
            for j in range(n_pts_y):
                idx = i + j*n_pts_x

                A[idx, idx] = stencil[1,1]
                has_left = (i>0)
                has_right = (i<n_pts_x-1)
                has_down = (j>0)
                has_up = (j<n_pts_y-1)

                # NSEW connections
                if has_up:
                    A[idx, idx + n_pts_x] = stencil[0, 1]
                if has_down:
                    A[idx, idx - n_pts_x] = stencil[2, 1]
                if has_left:
                    A[idx, idx - 1] = stencil[1, 0]
                if has_right:
                    A[idx, idx + 1] = stencil[1, 2]

                # diagonal connections
                if has_up and has_left:
                    A[idx, idx + n_pts_x - 1] = stencil[0, 0]
                if has_up and has_right:
                    A[idx, idx + n_pts_x + 1] = stencil[0, 2]
                if has_down and has_left:
                    A[idx, idx - n_pts_x - 1] = stencil[2, 0]
                if has_down and has_right:
                    A[idx, idx - n_pts_x + 1] = stencil[2, 2]
        A = A.tocsr()

        return Grid(A, grid_x)
