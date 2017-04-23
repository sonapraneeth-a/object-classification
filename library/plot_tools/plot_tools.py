# Source:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

import matplotlib.pyplot as plt
import seaborn as sns
from bokeh import mpl
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.models import HoverTool, BoxZoomTool, ResetTool, WheelZoomTool, PanTool, SaveTool
from bokeh.io import output_notebook
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_test, y_pred, classes=[], normalize=False, fig_size=(8,6),
                          title='Confusion matrix', cmap=plt.cm.Blues, plot_lib='matplotlib', matplotlib_style='default'):
    cm = confusion_matrix(y_test, y_pred)
    if len(classes) == 0:
        classes = list(range(cm.shape[0]))
    print('Confusion matrix, without normalization')
    print(cm)
    if normalize:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(norm_cm)
    if plot_lib == 'matplotlib':
        plt.style.use(matplotlib_style)
        plt.figure(figsize=fig_size)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=60)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    if plot_lib == 'seaborn':
        cm_df = pd.DataFrame(data=cm, index=classes, columns=classes)
        plt.figure(figsize=fig_size)
        ax = plt.axes()
        sns.heatmap(cm_df, ax=ax, annot=True, fmt='d')
        ax.set_title(title)
        plt.xticks(rotation=60)
        sns.plt.show()


def plot_variance(scores, means, stds, legend=['Legend1', 'Legend2'], colors=['blue', 'green'],
                  plot_title=['Title1', 'Title2'],
                  plot_xlabel=['X'], plot_ylabel=['Y'], plot_lib='matplotlib', matplotlib_style='default',
                  type='fill', fig_size=(8,6), bokeh_notebook=False):
    if plot_lib == 'matplotlib':
        plt.style.use(matplotlib_style)
        plt.grid()
        for i in range(means.shape[1]):
            if type == 'errorbar':
                plt.errorbar(np.arange(len(means[:,i])), means[:,i], stds[:,i])
            if type == 'fill':
                plt.fill_between(np.arange(len(means[:,i])), means[:,i] - stds[:,i], means[:,i] + stds[:,i],
                                 alpha=0.1, color='b')
            plt.plot(np.arange(len(means[:,i])), means[:,i], 'o-', color='b', label='Training score')
        plt.title(plot_title)
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.legend(loc='best')
        plt.axis('tight')
        plt.show()
    elif plot_lib == 'seaborn':
        fig = plt.figure(figsize=fig_size)
        plt.grid()
        sns_plot = sns.tsplot(data=scores, err_style=['ci_band', 'ci_bars'], marker='o', legend=True)
        sns_plot.set(xlabel=plot_xlabel, ylabel=plot_ylabel)
        sns.plt.title(plot_title)
        if bokeh_notebook is True:
            output_notebook()
        show(mpl.to_bokeh(fig))
    elif plot_lib == 'bokeh':
        if bokeh_notebook is True:
            output_notebook()
        plot_fig = []
        num_exp = means.shape[1]
        num_splits = means.shape[0]
        for i in range(num_exp):
            source = ColumnDataSource(
                data=dict(
                    x=np.arange(num_splits),
                    y=means[:, i].T,
                    z=stds[:, i].T,
                )
            )
            hover = HoverTool(
                tooltips=[
                    ("(x)", "($x)"),
                    ("(mean)", "($y)"),
                    ("(std)", "(@z)"),
                ]
            )
            p = figure(title='errorbars with bokeh', width=600, height=400,
                       tools=[hover, PanTool(), BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool()])
            p.xaxis.axis_label = plot_xlabel[i]
            p.yaxis.axis_label = plot_ylabel[i]
            p.title.text = plot_title[i]
            p.background_fill = 'beige'
            p.circle(np.arange(num_splits), means[:, i].T, color=colors[i], size=5, line_alpha=0, source=source)
            p.line(np.arange(num_splits), means[:, i].T, color=colors[i], legend=legend[i], source=source)
            plot_fig.append(p)
        for i in range(num_exp):
            for x, y, yerr in zip(np.arange(num_splits), means[:, i].T, stds[:, i].T):
                err_xs = []
                err_ys = []
                err_xs.append((x, x))
                err_ys.append((y - yerr, y + yerr))
            band_x = np.append(np.arange(num_splits), np.arange(num_splits)[::-1])
            band_y = np.append(means[:, i] - stds[:, i], (means[:, i] + stds[:, i])[::-1])
            plot_fig[i].multi_line(err_xs, err_ys, color=colors[i],line_width=1)
            plot_fig[i].patch(band_x, band_y, color='#7570B3', fill_alpha=0.1, line_width=0.2)
        grid = gridplot(plot_fig, ncols=2)
        show(grid)


def plot_accuracy(scores, legend=['Legend1', 'Legend2'], colors=['blue', 'green'], plot_title='Title',
                  plot_xlabel='X', plot_ylabel='Y', plot_lib='matplotlib', matplotlib_style='default',
                  fig_size=(800,600), bokeh_notebook=False):
    if plot_lib == 'matplotlib':
        plt.style.use(matplotlib_style)
        plt.grid()
        for i in range(scores.shape[0]):
            plt.plot(np.arange(len(scores[:,i])), scores[:,i], 'o-', color=colors[i], label=legend[i])
        plt.title(plot_title)
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.legend(loc='best')
        plt.axis('tight')
        plt.show()
    elif plot_lib == 'seaborn':
        fig = plt.figure(figsize=fig_size)
        plt.grid()
        sns_plot = sns.tsplot(data=scores, err_style=['ci_band', 'ci_bars'], marker='o', legend=True)
        sns_plot.set(xlabel=plot_xlabel, ylabel=plot_ylabel)
        sns.plt.title(plot_title)
        if bokeh_notebook is True:
            output_notebook()
        show(mpl.to_bokeh(fig))
    elif plot_lib == 'bokeh':
        if bokeh_notebook is True:
            output_notebook()
        num_exp = scores.shape[1]
        num_splits = scores.shape[0]
        hover = HoverTool(
            tooltips=[
                ("(x, accuracy)", "($x, $y)"),
            ]
        )
        p = figure(title=plot_title, width=fig_size[1], height=fig_size[0],
                   tools=[hover, PanTool(), BoxZoomTool(), ResetTool(), WheelZoomTool(), SaveTool()])
        p.xaxis.axis_label = plot_xlabel
        p.yaxis.axis_label = plot_ylabel
        p.title.text = plot_title
        p.background_fill = 'beige'
        for i in range(num_exp):
            source = ColumnDataSource(
                data=dict(
                    x=np.arange(num_splits),
                    y=scores[:, i],
                )
            )
            p.circle(np.arange(num_splits), scores[:, i], color=colors[i], size=5, line_alpha=0, source=source)
            p.line(np.arange(num_splits), scores[:, i], color=colors[i], legend=legend[i], source=source)
        show(p)
