import networkx as nx
import torch
import torch_geometric as tg
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.spatial as spat
import pyamg
import matplotlib.pyplot as plt
import matplotlib
import os

import pyamg.gallery.mesh
import pyamg.gallery.fem

import shapely.geometry as sg
from shapely.ops import unary_union
import ns.lib.sparse
import ns.lib.helpers

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
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=nx_data.edge_attr.float())

def graph_from_matrix_basic(A):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    nx_data = tg.utils.from_networkx(G, None, ['weight'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))

class Grid():
    def __init__(self, A_csr, x=None, extra={}):
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
        self.extra = extra

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
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
        if self.x is None:
            positions = None
        else:
            positions = {}
            for node in graph.nodes:
                positions[node] = self.x[node]

        nx.drawing.nx_pylab.draw_networkx(graph, ax=ax, pos=positions, arrows=False, with_labels=False, node_size=100)

    def plot_spider_agg(self, AggOp, P, ax=None, lw=4):
        '''
        Creates a spider plot, drawing lines from geometric aggregate centers to each node in an aggregate.
        It's recommended to combine this with plot_agg to get the aggregate borders as well.

        Parameters
        ----------
        AggOp : CSR sparse matrix
          n x nagg encoding of the aggregates AggOp[i,j] == 1 means node i is in aggregate j
        P : CSR sparse matrix
          n x nagg interpolation operator that is used for the edge opacities.
        ax : axis
          matplotlib axis
        lw : float
          line width
        '''

        if ax is None:
            ax = plt.gca()

        x_centers = ns.lib.sparse.col_normalize_csr(P, ord=1).T @ self.x
        cmap = matplotlib.cm.get_cmap('tab10')

        P = np.abs(P)
        for i in range(P.shape[1]):
            agg = P[:,i]
            P_max = spla.norm(agg, np.inf) # compute per-agg max interpolation weight, because aggs may have different scale
            nonzeros = agg.nonzero()[0]
            xc = x_centers[i]
            for j in nonzeros:
                Pv = agg[j, 0]
                xv = self.x[j]
                line = matplotlib.lines.Line2D([xc[0], xv[0]], [xc[1], xv[1]], color=cmap(i%cmap.N), alpha=Pv/P_max, linewidth=lw)
                ax.add_line(line)

        ax.plot(x_centers[:,0], x_centers[:,1], 'r*', markersize=6)


    def plot_agg(self, AggOp, ax=None, color=None, edgecolor='k', lw=1, alpha=0.7):
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

            todraw = unary_union(todraw)                    # union all objects in the aggregate
            todraw = todraw.buffer(0.04)                       # expand to smooth
            todraw = todraw.buffer(-0.02)                      # then contract

            try:
                xs, ys = todraw.exterior.xy                    # get all of the exterior points
                ax.fill(xs, ys, clip_on=False, alpha=alpha)
                ax.plot(xs, ys, color=edgecolor)
            except:
                pass                                           # don't plot singletons

    def save(self, fname):
        if not '.grid' in fname:
            fname = fname + '.grid'
            
        ns.lib.helpers.pickle_save_bz2(fname, {
            'A': self.A,
            'x': self.x,
            'extra': self.extra
        })

    def load(fname):
        if not '.grid' in fname:
            fname = fname + '.grid'

        loaded = ns.lib.helpers.pickle_load_bz2(fname)
        return Grid(loaded['A'], loaded['x'], loaded['extra'] if 'extra' in loaded else {})

    def load_dir(directory):
        grids = []
        for f in os.listdir(directory):
            if not '.grid' in f.lower():
                continue
            grids.append(Grid.load(os.path.join(directory, f)))
        return grids


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


    def meshio_2d_poisson_dirichlet(mesh, epsilon=1.0, theta=0.0):
        '''
        Creates a 2D poisson system on an unstructured mesh, defined by a meshio object

        Parameters
        ----------
        mesh : meshio.Mesh
          Mesh that defines the PDE domain.  Should have the following cells:
          - 'line' (for boundary data)
          - 'triangle' (for element data)
        epsilon : float
          Scaling of y-dimension for anisotropic problems
        theta : float
          Rotation of diffusion for anisotropic problems
        '''

        # Set up diffusion coefficient
        def kappa(x, y):
            c, s = np.cos(theta), np.sin(theta)
            Q = np.array([
                [c, -s],
                [s, c]
            ])
            A = np.diag([1., epsilon])
            return Q@A@Q.T

        pts = mesh.points
        boundary_indices = set()
        for (a, b) in mesh.cells_dict['line']:
            boundary_indices.add(a)
            boundary_indices.add(b)
        boundary_indices = list(boundary_indices)

        interior_mask = np.ones(pts.shape[0], dtype=bool)
        interior_mask[boundary_indices] = False
        R = (sp.eye(pts.shape[0]).tocsr())[interior_mask]

        mesh = pyamg.gallery.fem.mesh(pts[:, :2], mesh.cells_dict['triangle'].astype(np.int64), degree=1)
        A, _ = pyamg.gallery.fem.gradgradform(mesh, kappa=kappa, degree=1)
        A = A.tocsr()
        A_d = R@A@R.T
        A_d.eliminate_zeros()

        return Grid(A_d, (R@pts)[:, :2], {
            'epsilon': epsilon,
            'theta': theta
        })
    
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
        epsilon : float
          Scaling of y-dimension for anisotropic problems
        theta : float
          Rotation of diffusion for anisotropic problems

        Returns
        -------
        Grid object with given parameters.
        '''

        # Set up diffusion coefficient
        def kappa(x, y):
            c, s = np.cos(theta), np.sin(theta)
            Q = np.array([
                [c, -s],
                [s, c]
            ])
            A = np.diag([1., epsilon])
            return Q@A@Q.T

        # Structured mesh
        v, e = pyamg.gallery.mesh.regular_triangle_mesh(n_pts_x + 2, n_pts_y + 2)

        # Create boundary mask and boundary restriction operator, R
        boundary_mask = np.logical_or(
            np.logical_or(v[:, 0] == 0., v[:, 0] == 1.),
            np.logical_or(v[:, 1] == 0., v[:, 1] == 1.)
        )
        interior_mask = np.logical_not(boundary_mask)
        R = (sp.eye(v.shape[0]).tocsr())[interior_mask]

        # Update coordinates to lie within domain bounds
        v[:,0] = (v[:,0] + xdim[0]) * (xdim[1] - xdim[0])
        v[:,1] = (v[:,1] + ydim[0]) * (ydim[1] - ydim[0])

        # Discretize w/ linear finite elements
        mesh = pyamg.gallery.fem.mesh(v, e, degree=1)
        A, _ = pyamg.gallery.fem.gradgradform(mesh, kappa=kappa, degree=1)
        A = A.tocsr()
        A_d = R@A@R.T
        A_d.eliminate_zeros()

        return Grid(A_d, R@v, {
            'epsilon': epsilon,
            'theta': theta
        })

    def structured_2d_poisson_neumann(n_pts_x, n_pts_y,
                                      xdim=(0,1), ydim=(0,1),
                                      epsilon=1.0, theta=0.0):
        '''
        Creates a 2D poisson system on a structured grid, discretized using finite elements.
        Homogeneous neumann boundary conditions are assumed.

        Parameters
        ----------
        n_pts_x : integer
          Number of points in the x dimension (including boundary points)
        n_pts_y : integer
          Number of points in the y dimension (including boundary points)
        xdim : tuple (float, float)
          Bounds for domain in x dimension.  Represents smallest and largest x values.
        ydim : tuple (float, float)
          Bounds for domain in y dimension.  Represents smallest and largest y values.

        Returns
        -------
        Grid object with given parameters.
        '''

        # Set up diffusion coefficient
        def kappa(x, y):
            c, s = np.cos(theta), np.sin(theta)
            Q = np.array([
                [c, -s],
                [s, c]
            ])
            A = np.diag([1., epsilon])
            return Q@A@Q.T

        # Structured mesh
        v, e = pyamg.gallery.mesh.regular_triangle_mesh(n_pts_x, n_pts_y)

        # Update coordinates to lie within domain bounds
        v[:,0] = (v[:,0] + xdim[0]) * (xdim[1] - xdim[0])
        v[:,1] = (v[:,1] + ydim[0]) * (ydim[1] - ydim[0])

        # Discretize w/ linear finite elements
        mesh = pyamg.gallery.fem.mesh(v, e, degree=1)
        A, _ = pyamg.gallery.fem.gradgradform(mesh, kappa=kappa, degree=1)
        A = A.tocsr()

        return Grid(A, v)
