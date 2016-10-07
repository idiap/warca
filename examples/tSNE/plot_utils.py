"""
 *  Some plot utilities 
 *  
 *  Copyright (c) 2016 Idiap Research Institute, http://www.idiap.ch/
 *  Written by Cijo Jose <cijose@idiap.ch>
 *
 *  This is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 *  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with selector.  If not, see <http://www.gnu.org/licenses/>.
 *

"""
import colorsys
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

def distinguishable_colors(num_colors):
    colors = np.zeros((num_colors, 3))
    for i,ii in enumerate(np.arange(0., 360., 360./num_colors)):
	hue = ii/360.
        lightness = (50 + 20.*(i%2 == 0)) / 100.
	saturation = (100 -50 * (i % 3 == 0)) / 100.
        colors[i, :] = colorsys.hls_to_rgb(hue, lightness, saturation)
    return colors


def plot_dataset(X, labels, save_dir = './', name = 'plot', title = 'Scatter plot'):
    X -= np.min(X)
    X /= np.max(X)    
    labels = labels.astype(int)
    ul = np.unique(labels)
    colors = distinguishable_colors(ul.size)
    sc = []
    plt.figure(figsize=(25, 25))
    for (i,label) in enumerate(ul):
        idx = np.where(labels == label)[0]
        sc.append(plt.scatter(X[idx, 0], X[idx, 1], color=colors[i, :]))

    for i  in range(labels.size):
        plt.annotate(labels[i], (X[i, 0],X[i, 1]), fontsize=7)

    plt.title(title)
    plt.savefig(save_dir + '/'+name+'.pdf', format='pdf', dpi=1000)
    #plt.show()

