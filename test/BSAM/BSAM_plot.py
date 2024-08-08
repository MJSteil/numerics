## @file BSAM_plot.py
#  Basic plotting routines for 2D BSAM grid (Mandelbrot example)

import numpy as np
import pandas as pd
import matplotlib
import json

matplotlib.use('Agg') # run without frontend https://stackoverflow.com/a/4935945/
from matplotlib import pyplot as plt
from pylab import rcParams


#[https://stackoverflow.com/a/18926541/6644522]
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap =  matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

name="mandelbrot_lvl_"
nv=['0','1','2','3','4']
for n in nv:
    plt.style.use('classic')
    
    fontsize=12
    fontsize_small=10
    rcParams['figure.figsize'] = 10, 10
    rcParams['legend.fontsize'] = fontsize_small
    rcParams['font.family'] = 'sans-serif'
    rcParams['mathtext.fontset'] = 'dejavusans' #CM
    #rcParams['text.usetex'] = True
    #rcParams['text.latex.preview'] = True #https://stackoverflow.com/a/42732444
    
    fig, ax = plt.subplots()
    
    # Load the JSON file
    with open('mandelbrot_lvl_0.json', 'r') as file:
        data = json.load(file)
    
    points=np.array(data['points'])

    c_map=truncate_colormap(plt.get_cmap('viridis'),0.0,1.0)
    density=ax.tripcolor(points[::,0],points[::,1],-np.log(points[::,2]/np.max(points[::,2])), edgecolor='white', linewidths=.05,shading='flat',cmap=c_map,rasterized=False) #
    
    plt.xlabel("$\mathrm{Re}(c)$",fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    
    plt.ylabel("$\mathrm{Im}(c)$",fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    plt.title("Mandelbrot set ($z_0=0,\\,z_{n+1}=z_n^2+c$) number of iterations - BSAM up to level "+n,pad=20)
    
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(name+n+'.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600,facecolor='w', edgecolor='w')
    plt.clf()
    plt.close()
    #plt.show()
    
