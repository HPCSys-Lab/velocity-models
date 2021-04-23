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


#reduce_model("overthrust_3D_true_model.h5","reduced_overthrust_3D_true_model.h5")

infilename="overthrust_3D_true_model.h5"

with h5py.File(infilename, 'r') as infile:
    print("Input: ",infilename)
    print("   Number of grid points   n= %s  " % infile['n'][()])
    print("   Origin                  o= %s  " % infile['o'][()])
    print("   Distances               d= %s  " % infile['d'][()])
    # velocity model data
    in_m = infile['m'][()]
    in_n = infile['n'][()]
    in_o = infile['o'][()]
    in_d = infile['d'][()]
    in_copytight = infile['copyright'][()]

    z0 = 0
    z1 = 201
    x0 = 50
    x1 = 451
    y0 = 500
    y1 = 801
    #out_m = cut_model(in_m,0,200,50,450,500,800)
    out_m = cut_model(in_m,z0,z1,x0,x1,y0,y1)
    out_n = np.array([x1-x0, y1-y0, z1-z0])
    out_o = in_o
    out_d = in_d
    out_copytight = in_copytight
    outfilename="reduced_overthrust_3D_true_model.h5"

    with h5py.File(outfilename, 'w') as outfile:
        outfile.create_dataset('m', data=out_m)
        outfile.create_dataset('n', data=out_n)
        outfile.create_dataset('o', data=out_o)
        outfile.create_dataset('d', data=out_d)
        outfile.create_dataset('copyright', data=out_copytight)
        print("Output: ",outfilename)
        print("   Number of grid points   n= %s  " % out_n)
        print("   Origin                  o= %s  " % out_o)
        print("   Distances               d= %s  " % out_d)
