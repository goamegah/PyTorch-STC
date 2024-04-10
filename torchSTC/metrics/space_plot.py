import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import umap


import warnings
import colorsys


class SpacePlot:
    def commonSpace_plot(self, B, comp = [0, 1], tagLabels = None, 
                         data_name = "", dimred: Union[Literal['UMAP', 'TSNE']] = 'UMAP'):

        # Apply UMAP on B?
        my_reduction = ""
        embedding = None
        if dimred == 'UMAP':
            reducer = umap.UMAP(n_components = max(comp)+1)
            embedding = reducer.fit_transform(B)
            my_reduction = "with UMAP"
        elif dimred == 'TSNE':
            reducer = TSNE(n_components = max(comp)+1)
            embedding = reducer.fit_transform(B)
            my_reduction = "with TSNE"
        else:
            embedding = B

        colLabels = [0]*embedding.shape[0]
        if tagLabels is not None:
            colLabels = [int(i) for i in tagLabels]

        if len(comp) == 2:
            self._commonSpace_2D(embedding, comp = comp, colLabels = colLabels, data_name = data_name, reduction = my_reduction)

        elif len(comp) == 3:
            self._commonSpace_3D(embedding, comp = comp, colLabels = colLabels, data_name = data_name, reduction = my_reduction)

        else:
            warnings.warn("The number of components should be 2 or 3.")
            

    def _commonSpace_2D(self, embedding, comp = [0, 1], colLabels = None, data_name = "", reduction = ""):

        assert len(comp) == 2

        colors = self._get_colors(20)
        if colLabels is not None:
            colors = self._get_colors(len(set(colLabels)))
            
        plt.scatter(embedding[:, comp[0]], embedding[:, comp[1]],
                    c = np.array(colors)[colLabels],
                    s = 3, edgecolors = "black", linewidth = 0.25,
                    marker = 'o')

        plt.title("{} Self-training embeddings {} ({}|{})".format(data_name, reduction, comp[0], comp[1]))
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

        if len(reduction) == 0:
            plt.axis((-1.1, 1.1, -1.1, 1.1))

    def _commonSpace_3D(self, embedding, comp = [0, 1, 2], colLabels = None, data_name = "", reduction = ""):

        assert len(comp) == 3

        fig = plt.figure()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        colors = self._get_colors(20)
        if colLabels is not None:
            colors = self._get_colors(len(set(colLabels)))
        ax.scatter(embedding[:,comp[0]], embedding[:,comp[1]], embedding[:,comp[2]],
                       c = np.array(colors)[colLabels],
                       s = 3, edgecolors = "black", linewidth = 0.25,
                   marker = 'o')
        plt.title("{} Self-training embeddings {} ({}|{}|{})".format(data_name, reduction, comp[0], comp[1], comp[2]))

        if len(reduction) == 0:
            plt.axis((-1.1, 1.1, -1.1, 1.1))

    def _get_colors(self, num_colors):
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return colors
        
