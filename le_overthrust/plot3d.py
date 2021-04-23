import numpy as np
import pyvista as pv
import h5py

def read_model(filename):
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())

        # Get the data
        data = list(f['m'])
        data = np.array(data)
        data = (1 / (data ** (1 / 2)))

        return data


def plot(d):
    data = pv.wrap(d)
    data.plot(volume=True)

#data = read_model('overthrust_3D_true_model.h5')
data = np.random.randint(0, 2, size=(3,3,3))

plot(data)
