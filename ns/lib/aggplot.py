import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.cm
import mpl_toolkits.mplot3d.art3d
import mpl_toolkits.mplot3d.axes3d

import skimage.measure

def _plot_thread(pipe):
    plt.ion()
    plt.figure()
    plt.show()

    while True:
        while pipe.poll():
            data = pipe.recv()
            if data[0] == 'exit':
                break
            elif data[0] == 'run':
                data[1](*data[2], **data[3])
            elif data[0] == 'axes_run':
                if not isinstance(plt.gca(), matplotlib.axes.Axes):
                    plt.axes()
                getattr(plt.gca(), data[1])(*data[2], **data[3])
            elif data[0] == 'axes3d_run':
                if not isinstance(plt.gca(), mpl_toolkits.mplot3d.axes3d.Axes3D):
                    plt.axes(projection='3d')
                getattr(plt.gca(), data[1])(*data[2], **data[3])
        plt.gcf().canvas.start_event_loop(0.5)


def plot_agg_3d(grid, AggOp):
    agg_assign = np.array(AggOp.argmax(axis=1)).flatten()
    colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    agg_colors = colors[agg_assign % len(colors)].tolist()
    plt.gca().scatter3D(grid.x[:,0], grid.x[:,1], grid.x[:,2], c=agg_colors)


def plot_3d_grid_voxel(grid, u):
    xs = np.sort(grid.x[:,0])
    ys = np.sort(grid.x[:,1])
    zs = np.sort(grid.x[:,2])

    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    dz = zs[1]-zs[0]

    Nx = len(np.unique(xs))
    Ny = len(np.unique(ys))
    Nz = len(np.unique(zs))

    xx, yy, zz = np.meshgrid(np.linspace(xs[0]-dx/2, xs[-1]+dx/2, Nx+1),
                             np.linspace(ys[0]-dy/2, ys[-1]+dy/2, Ny+1),
                             np.linspace(zs[0]-dz/2, zs[-1]+dz/2, Nz+1))

    sort = np.lexsort((grid.x[:,2], grid.x[:,1], grid.x[:,0]))

    sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap())
    col = sm.to_rgba(u[sort])
    plt.gca().voxels(xx, yy, zz, np.ones((Nx, Ny, Nz)), facecolors=col.reshape((Nx, Ny, Nz, 4)), shade=False)
    plt.colorbar(mappable=sm)


def plot_agg_3d_voxel(grid, AggOp, opacity=0.5):
    agg_assign = np.array(AggOp.argmax(axis=1)).flatten()
    colors_rgb = np.array([
        [31, 119, 180],  #1f77b4
        [255, 127, 14],  #ff7f0e
        [44, 160, 44],   #2ca02c
        [214, 39, 40],   #d62728
        [184, 103, 189], #9467bd
        [140, 86, 75],   #8c564b
        [227, 119, 194], #e377c2
        [127, 127, 127], #7f7f7f
        [188, 189, 34],  #bcbd22
        [23, 190, 207],  #17becf
    ], dtype=np.float64)
    colors_rgb /= 255.
    colors_rgba = np.column_stack((colors_rgb, np.ones(colors_rgb.shape[0]) * opacity))

    xs = np.sort(grid.x[:,0])
    ys = np.sort(grid.x[:,1])
    zs = np.sort(grid.x[:,2])

    dx = xs[1]-xs[0]
    dy = ys[1]-ys[0]
    dz = zs[1]-zs[0]

    Nx = len(np.unique(xs))
    Ny = len(np.unique(ys))
    Nz = len(np.unique(zs))

    xx, yy, zz = np.meshgrid(np.linspace(xs[0]-dx/2, xs[-1]+dx/2, Nx+1),
                             np.linspace(ys[0]-dy/2, ys[-1]+dy/2, Ny+1),
                             np.linspace(zs[0]-dz/2, zs[-1]+dz/2, Nz+1))

    sort = np.lexsort((grid.x[:,2], grid.x[:,1], grid.x[:,0]))
    AggOp = AggOp[sort]

    ax = plt.gca()
    for agg in range(AggOp.shape[1]):
        Agg = np.array(AggOp[:, agg].todense(), dtype=bool).flatten()
        col = colors_rgba[agg % colors_rgba.shape[0]]
        ax.voxels(xx, yy, zz, Agg.reshape((Nx, Ny, Nz)), facecolors=col)
        # verts, faces, normals, values = skimage.measure.marching_cubes(Agg.reshape((Nx, Ny, Nz)), mask=(Agg==1).reshape((Nx,Ny,Nz)))
        # mesh = mpl_toolkits.mplot3d.art3d.Poly3DCollection(verts[faces])
        # mesh.set_edgecolor('k')
        # ax.add_collection3d(mesh)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def create_axes(*args, **kwargs):
    plt.axes(*args, **kwargs)


def clear():
    for image in plt.gca().images:
        image.colorbar.remove()
    plt.clf()


def colorbar(*args, **kwargs):
    plt.colorbar()


def scatter3D_wrapper(*args, **kwargs):
    if 'colorbar' in kwargs:
        colorbar = kwargs['colorbar']
        del kwargs['colorbar']
        path = plt.gca().scatter3D(*args, **kwargs)
        if colorbar:
            plt.colorbar(path, ax=plt.gca())
    else:
        plt.gca().scatter3D(*args, **kwargs)


plot_funcs = {
    'plot_agg_3d': plot_agg_3d,
    'plot_3d_grid_voxel': plot_3d_grid_voxel,
    'plot_agg_3d_voxel': plot_agg_3d_voxel,
    'scatter3D': scatter3D_wrapper,
    'create_axes': create_axes,
    'clear': clear,
    'colorbar': colorbar
}


def rpc_wrapper(conn, func):
    def f(*args, **kwargs):
        conn.send(('run', func, args, kwargs))
    return f


def rpc_axes_wrapper(conn, funcname):
    def f(*args, **kwargs):
        conn.send(('axes_run', funcname, args, kwargs))
    return f


def rpc_axes3d_wrapper(conn, funcname):
    def f(*args, **kwargs):
        conn.send(('axes3d_run', funcname, args, kwargs))
    return f


class ThreadedPlot:
    '''
    Class that wraps a matplotlib axes object in a separate process, allowing
    the main thread/process to do calculations while keeping the figure window responsive.
    '''

    def __init__(self):
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        mp_ctx = multiprocessing.get_context('spawn')
        self.process = mp_ctx.Process(target=_plot_thread, args=(self.child_conn,))
        self.process.start()

        # Add custom plot functions
        for k, v in plot_funcs.items():
            setattr(self, k, rpc_wrapper(self.parent_conn, v))

        # Add matplotlib axes functions
        for func in dir(matplotlib.axes.Axes):
            if func[0] == '_':
                continue
            setattr(self, func, rpc_axes_wrapper(self.parent_conn, func))

        # Add matplotlib 3d axes functions
        for func in dir(mpl_toolkits.mplot3d.axes3d.Axes3D):
            if func[0] == '_' or hasattr(self, func):
                continue
            setattr(self, func, rpc_axes3d_wrapper(self.parent_conn, func))

    def __del__(self):
        self.parent_conn.send(('exit',))
        self.process.join()
