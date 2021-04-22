import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils_claus import *
#from argparse import ArgumentParser

def cut_model(data, z1, z2, x1, x2, y1, y2):
    new_data = data[z1:z2,x1:x2,y1:y2]
    return new_data

def reduce_model(infilename,outfilename):
    with h5py.File(infilename, 'r') as infile:
        data_m = infile['m'][()]
        m_reduced = cut_model(data_m,0,200,50,450,500,800)
        data_o = infile['o'][()]
        data_d = infile['d'][()]
        #data_copytight = infile['copyright'][()]
        o_reduced = data_o
        d_reduced = data_d

        with h5py.File(outfilename, 'w') as outfile:
            outfile.create_dataset('m', data=m_reduced)
            outfile.create_dataset('o', data=o_reduced)
            outfile.create_dataset('d', data=d_reduced)


reduce_model("overthrust_3D_true_model.h5","reduced_overthrust_3D_true_model.h5")
