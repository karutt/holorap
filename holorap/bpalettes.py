import bokeh.palettes as c
import numpy as np


def color_lerp(num, type="magma"):
    """
    ["grey", "cividis", "inferno", "magma", "plasma", "viridis", "cividis"]
    """
    return eval("c." + type)(num)


def color_lerp_list():
    return ["grey", "cividis", "inferno", "magma", "plasma", "viridis", "cividis"]


def int_to_color(x, type="Magma"):
    "'YlGn','YlGnBu','GnBu','BuGn','PuBuGn','PuBu','BuPu','RdPu','PuRd','OrRd','YlOrRd','YlOrBr','Purples','Blues','Greens','Oranges','Reds','Greys','PuOr','BrBG','PRGn','PiYG','RdBu','RdGy','RdYlBu','Spectral','RdYlGn','Accent','Dark2','Paired','Pastel1','Pastel2','Set1','Set2','Set3','Category10','Category20','Category20b','Category20c','Colorblind','Magma','Inferno','Plasma','Viridis' ..etc"
    x = np.array(x)
    N = x.max()
    col = np.array([None] * len(x)).astype("<U7")
    palett = eval("c." + type)[N + 1] if N > 2 else eval("c." + type)[3][::-1]
    for i in range(N + 1):
        col[i == x] = palett[i]
    return col


def color_list():
    return list(c.all_palettes.keys())
