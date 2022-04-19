import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.models import CrosshairTool
import os
import json
from bokeh.themes.theme import Theme


def fig(dw=600, dh=320, dark_theme=True, extension="bokeh"):

    hv.extension(extension)
    hv.opts.defaults(
        hv.opts.Scatter(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True),
        hv.opts.Rectangles(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True,  line_color="None"),
        hv.opts.Histogram(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True),
        hv.opts.Contours(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True),
        hv.opts.Text(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True),
        hv.opts.Area(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True, fill_alpha=0.6),
        hv.opts.Curve(height=dh, width=dw, active_tools=[
            "wheel_zoom"], shared_axes=True, show_grid=True),
    )

    if dark_theme:
        with open(os.path.dirname(__file__)+"/dark_theme.json", "r") as dt:
            json_dark_theme = json.load(dt)
        hv.renderer('bokeh').theme = Theme(json=json_dark_theme)
    else:
        with open(os.path.dirname(__file__)+"/white_theme.json", "r") as dt:
            json_white_theme = json.load(dt)
        hv.renderer('bokeh').theme = Theme(json=json_white_theme)
    return hv
