import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils_claus import *

def plot_velocity_model(model, file_name="velocity_model",
                        colorbar=True, cmap="jet", show=False):

    # create the destination dir
    os.makedirs("plots", exist_ok=True)

    # process data and generate the plot
    plot = plt.imshow(model, cmap=cmap)

    # Create aligned colorbar on the right
    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plot, cax=cax)

    plt.savefig("plots/{}.png".format(file_name), format="png")

    if show:
        plt.show()

    plt.close()

    print("Velocity model saved in plots/{}.png".format(file_name))

def read_model(filename):
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())

        # Get the data
        data = list(f['m'])

        data = np.array(data)
        # convert to m/s
        #data = (1 / (data ** (1 / 2))) * 1000.0
        data = (1 / (data ** (1 / 2)))

        return data

# import matplotlib.pyplot as plt
# filename = "overthrust_2D_initial_model.h5"
#filename = "overthrust_3D_initial_model.h5"
filename = "overthrust_3D_true_model.h5"

data = read_model(filename)
print(data.shape)

plot_velocity_model (data[:,:,500], file_name="slice_dim3",show=True)
plot_velocity_model (data[:,400,:], file_name="slice_dim2",show=True)
plot_velocity_model (data[100,:,:], file_name="slice_dim1",show=True)

filename = "reduced_overthrust_3D_true_model.h5"

data = read_model(filename)

plot_velocity_model (data[:,:,0], file_name="slice_dim3_cut",show=True)
plot_velocity_model (data[:,250,:], file_name="slice_dim2_cut",show=True)
plot_velocity_model (data[100,:,:], file_name="slice_dim1_cut",show=True)
