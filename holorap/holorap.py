import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.models import CrosshairTool
import os
import json
from bokeh.themes.theme import Theme


class Holorap:
    def __init__(self, dark_theme=False, dw=600, dh=320, extension="bokeh"):
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
        self.figs = []
        self.ch = CrosshairTool(line_color="#b5c0c2")
        self.count = 1
        self.info = {"type": [], "auto_scale": []}
        if dark_theme:
            with open(os.path.dirname(__file__)+"/dark_theme.json", "r") as dt:
                json_dark_theme = json.load(dt)
            hv.renderer('bokeh').theme = Theme(json=json_dark_theme)
        else:
            with open(os.path.dirname(__file__)+"/white_theme.json", "r") as dt:
                json_white_theme = json.load(dt)
            hv.renderer('bokeh').theme = Theme(json=json_white_theme)

    def set_label(self, xlabel, ylabel, dtype):
        if xlabel == None:
            xlabel = dtype + "_x" + str(self.count)
        if ylabel == None:
            ylabel = dtype + "_y" + str(self.count)
        self.count += 1
        return xlabel, ylabel

    def set_crosshair(self, f):
        f.opts(hooks=[
            lambda plot, _:
                plot.state.add_tools(self.ch)
        ])

    def line(self, x, y=None, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "line")
        data = (x, y) if type(y) != type(None) else pd.Series(x, name=ylabel)
        f = hv.Curve(data, xlabel, ylabel).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)
        self.info["auto_scale"].append([asc])
        self.info["type"].append(["line"])
        self.figs.append(f)

    def add_line(self, x, y=None, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "line")
        data = (x, y) if type(y) != type(None) else pd.Series(x, name=ylabel)
        f = hv.Curve(data, xlabel, ylabel, label=label).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("line")
        self.figs[-1] *= f

    def segment(self, x0, y0, x1, y1, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "segment")
        f = hv.Segments((x0, y0, x1, y1), kdims=[
                        xlabel, ylabel, "a", "b"]).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)
        self.info["auto_scale"].append([asc])
        self.info["type"].append(["segment"])
        self.figs.append(f)

    def add_segment(self, x0, y0, x1, y1, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "segment")
        f = hv.Segments((x0, y0, x1, y1), kdims=[
                        xlabel, ylabel, "a", "b"]).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("segment")
        self.figs[-1] *= f

    def scatter(self, x, y=None, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "scatter")
        data = (x, y) if type(y) != type(
            None) else pd.Series(x, name="scatter_y")
        f = hv.Scatter(data, xlabel, ylabel, label=label).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)
        self.info["auto_scale"].append([asc])
        self.info["type"].append(["scatter"])
        self.figs.append(f)

    def add_scatter(self, x, y=None, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "scatter")
        data = (x, y) if type(y) != type(
            None) else pd.Series(x, name="scatter_y")
        f = hv.Scatter(data, xlabel, ylabel, label=label).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("scatter")
        self.figs[-1] *= f

    def histogram(self, y, bins=20, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "histogram")
        frequencies, edges = np.histogram(y, bins)
        f = hv.Histogram((edges[1:], frequencies), kdims=[
                         xlabel], vdims=[ylabel]).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)
        self.info["auto_scale"].append([asc])
        self.info["type"].append(["hist"])
        self.figs.append(f)

    def add_histogram(self, y, bins=20, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "histogram")
        frequencies, edges = np.histogram(y, bins)
        f = hv.Histogram((edges[1:], frequencies), kdims=[
                         xlabel], vdims=[ylabel]).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("hist")
        self.figs[-1] *= f

    def labels(self, x, y, t, **kwargs):
        """
        Style Options :
        angle, cmap, muted, text_align, text_alpha, text_baseline, text_color, text_font, text_font_size, text_font_style, visible
        """
        f = hv.Labels((x, y, t)).opts(**kwargs)
        self.info["auto_scale"][-1].append(False)
        self.info["type"][-1].append("label")
        self.figs[-1] *= f

    def area(self, x, y0=None, y1=None, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "area")
        data = (x, y) if type(y) != type(None) else pd.Series(x, name="area_y")
        f = None
        if type(y0) == type(None):
            f = hv.Area(x).opts(**kwargs)
        elif type(y1) == type(None):
            f = hv.Area(y0).opts(**kwargs)
        else:
            f = hv.Area((x, y0, y1), vdims=['y', 'y2']).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)
        self.info["auto_scale"].append([asc])
        self.info["type"].append(["area"])
        self.figs.append(f)

    def add_area(self, x, y0=None, y1=None, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "area")
        data = (x, y) if type(y) != type(None) else pd.Series(x, name="area_y")
        f = None
        if type(y0) == type(None):
            f = hv.Area(x).opts(**kwargs)
        elif type(y1) == type(None):
            f = hv.Area(y0).opts(**kwargs)
        else:
            f = hv.Area((x, y0, y1), vdims=['y', 'y2']).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("area")
        self.figs[-1] *= f

    def rect(self, x, y0, y1, bar_w, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "rect")
        bar_w *= 0.8
        x0 = x-bar_w/2
        x1 = x+bar_w/2
        f = hv.Rectangles((x0, y0, x1, y1), [
                          xlabel, ylabel, "x1", "y1"]).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)
        self.info["auto_scale"].append([asc])
        self.info["type"].append(["rect"])
        self.figs.append(f)

    def add_rect(self, x, y0, y1, bar_w, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "rect")
        bar_w *= 0.8
        x0 = x-bar_w/2
        x1 = x+bar_w/2
        f = hv.Rectangles((x0, y0, x1, y1), [
                          xlabel, ylabel, "x1", "y1"]).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("rect")
        self.figs[-1] *= f

    def bar(self, x, y=None, bar_w=None, bottom=0, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "bar")
        if type(y) == type(None):
            y = pd.Series(x, name="bar_y")
            x = np.arange(len(y))
            bar_w = 1
        bar_w *= 0.8
        x0 = x-bar_w/2
        x1 = x+bar_w/2
        y0 = pd.Series([bottom]*len(x))
        y1 = y
        f = hv.Rectangles((x0, y0, x1, y1), [
                          xlabel, ylabel, "x1", "y1"]).opts(**kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)

        self.info["auto_scale"].append([asc])
        self.info["type"].append(["bar"])
        self.figs.append(f)

    def add_bar(self, x, y=None, bar_w=None, bottom=0, label="", xlabel=None, ylabel=None, asc=False, **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "bar")
        if type(y) == type(None):
            y = pd.Series(x, name="bar_y")
            x = np.arange(len(y))
            bar_w = 0.8
        x0 = x-bar_w/2
        x1 = x+bar_w/2
        y0 = [bottom]*len(x)
        y1 = y

        f = hv.Rectangles((x0, y0, x1, y1), [
                          xlabel, ylabel, "x1", "y1"]).opts(**kwargs)
        self.info["auto_scale"][-1].append(asc)
        self.info["type"][-1].append("bar")
        self.figs[-1] *= f

    def contours(self, f,  levels=20, x_range=(-10, 10), y_range=(-10, 10), resolution=100, label="", xlabel=None, ylabel=None, cmap="blues", **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "contours")
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        xx, yy = np.meshgrid(x, y)
        z = f(xx, yy)
        img = hv.Image(z, kdims=[xlabel, ylabel], bounds=(
            x_range[0], y_range[0], x_range[1], y_range[1]))
        f = hv.operation.contours(img, levels=levels).opts(cmap=cmap, **kwargs)
        if "tools" in kwargs and "crosshair" in kwargs["tools"]:
            self.set_crosshair(f)

        self.info["auto_scale"].append([False])
        self.info["type"].append(["contours"])
        self.figs.append(f)

    def add_contours(self, f,  levels=20, x_range=(-10, 10), y_range=(-10, 10), resolution=100, label="", xlabel=None, ylabel=None, cmap="blues", **kwargs):
        xlabel, ylabel = self.set_label(xlabel, ylabel, "contours")
        x = np.linspace(*x_range, resolution)
        y = np.linspace(*y_range, resolution)
        xx, yy = np.meshgrid(x, y)
        z = f(xx, yy)
        img = hv.Image(z, kdims=[xlabel, ylabel], bounds=(
            x_range[0], y_range[0], x_range[1], y_range[1]))
        f = hv.operation.contours(img, levels=levels).opts(cmap=cmap, **kwargs)
        self.info["auto_scale"][-1].append(False)
        self.info["type"][-1].append(["contours"])
        self.figs[-1] *= f

    def candlesticks(self, candlesticks):
        x = candlesticks.open_time
        hr.segment(x[up], candlesticks.high[up], x[up],
                   candlesticks.low[up], line_color="green", auto_scale=True)
        hr.add_rect(x[up], candlesticks.open[up],
                    candlesticks.close[up], x[1] - x[0], fill_color="green")
        hr.add_segment(x[~up], candlesticks.high[~up], x[~up],
                       candlesticks.low[~up], line_color="red", auto_scale=True)
        hr.add_rect(x[~up], candlesticks.open[~up],
                    candlesticks.close[~up], x[1] - x[0], fill_color="red")

    def show(self, col=None):
        # -------auto_scaleの適応--------
        for i in range(len(self.figs)):
            if sum(self.info["auto_scale"][i]) != 0 and "contours" not in self.info["type"][i]:
                self.figs[i] = self.auto_scale(self.figs[i],
                                               self.info["auto_scale"][i],
                                               self.info["type"][i])
            if "contours" in self.info["type"][i]:
                def f(plot, a):
                    plot.state.legend.visible = False
                self.figs[i].opts(hooks=[f])
        # -------表示のレイアウトの設定--------
        if len(self.figs) == 1:
            return self.figs[0]
        else:
            if not col:
                col = len(self.figs)
            layout = self.figs[0]
            for fig in self.figs[1:]:
                layout += fig
            return layout.cols(col)

    def auto_scale(self, fig, auto_scales, fig_type):
        margin = 0.1
        # 最大最小を求めるnumpyデータ
        datas = self.get_fig_data(fig, fig_type)

        def draw(x_range, y_range):
            if x_range == None:
                return fig
            else:
                # x_range内での最大最小
                yr_start, yr_end = self.max_min_val_in_x_range(
                    x_range, datas, auto_scales)

                # 不都合が起こった場合は, 元の範囲のまま
                if yr_start == False or yr_start == yr_end or yr_start == None:
                    yr_start, yr_end = y_range
                else:
                    if "line" not in fig_type and "area" not in fig_type and "scatter" not in fig_type and "rect" not in fig_type and "segment" not in fig_type:
                        yr_start, yr_end = min(0, yr_start), max(0, yr_end)
                    ran = abs(yr_end - yr_start)
                    yr_start = yr_start - ran*margin
                    yr_end = yr_end + ran*margin

                return fig.opts(hooks=[
                    lambda plot, _:
                        plot.handles['y_range'].update(
                            start=yr_start, end=yr_end)
                ], active_tools=["wheel_zoom"])

        f = hv.DynamicMap(draw, streams=[hv.streams.RangeXY()])
        return f

    def get_fig_data(self, fig, fig_type):

        if str(type(fig.data)) == "<class 'collections.OrderedDict'>":
            datas = []
            for i, key in enumerate(fig.data):
                data = self.format_dtype(pd.DataFrame(
                    fig.data[key].data).to_numpy(), fig_type[i])
                datas.append(data)
            return datas
        else:
            data = self.format_dtype(pd.DataFrame(
                fig.data).to_numpy(), fig_type[0])
            return [data]

    def format_dtype(self, d, fig_type):
        if fig_type == "line" or fig_type == "scatter" or fig_type == "area" or fig_type == "hist":
            return d
        elif fig_type == "bar":
            return d[:, [0, 3]]
        elif fig_type == "rect" or fig_type == "segment":
            return d[:, [0, 1, 3]]

    def max_min_val_in_x_range(self, x_range, datas, auto_scales):
        max_values, min_values = [], []
        xr_start, xr_end = x_range
        xr_start = pd.Series([xr_start])[0]
        xr_end = pd.Series([xr_end])[0]
        for data, auto_scale in zip(datas, auto_scales):
            if auto_scale:
                x, y = data[:, 0], data[:, 1:]
                mask = (xr_start <= x) & (x <= xr_end)
                if mask.sum() == 0:
                    continue
                y_in_x_range = y[mask]
                max_values.append(np.nanmax(y_in_x_range))
                min_values.append(np.nanmin(y_in_x_range))

        if len(min_values) == 0 or len(max_values) == 0:
            return False, False
        else:
            return min(min_values), max(max_values)
