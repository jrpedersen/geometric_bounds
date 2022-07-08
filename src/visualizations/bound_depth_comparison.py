#import tensorboard as tb
#experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
import pathlib
import traceback

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

from experiments import export_metrics

lightning_logs =pathlib.Path("lightning_logs")
logs_models = list(lightning_logs.iterdir())
models_experiments = [path for model in logs_models for path in model.iterdir()]
#models_experiments = [models_experiments[0]]

metrics = ['train/train_acc_epoch',
           'validation/valid_acc_epoch',
           'bounds/Gradients_x',
           'bounds/Bound',
           'bounds/Gradients parameters',
           'bounds/Gradients_x times bound',
           'bounds/Gradients_x div Bound']

dataframes = {str(path) : export_metrics(str(path), metrics) for path in models_experiments}
#dataframes = next(iter(dataframes))


if 0:
    dictframes_all = {}
    for key, dictframes in dataframes.items():
        # TODO: Fix the concatenation when the step sizes differ.
        new_key='.'.join(key.split('\\')[-2:])
        dictframes_all[new_key] = pd.concat([v.rename(columns={'value': k}) for k,v in dictframes.items()],axis=1)

    # Choose one example to develop.
    name, df = next(iter(dictframes_all.items()))
    df = df.loc[:,~df.columns.duplicated()].copy()

    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)
    # accuracy
    ax_acc = fig.add_subplot(gs[3, :])
    ax_acc.plot(df['step'], df[metrics[0]], label= 'Training')
    ax_acc.plot(df['step'], df[metrics[1]], label= 'Validation')
    ax_acc.set_ylim([0.95,1.0])
    ax_acc.grid(True)
    ax_acc.set_xlabel('Steps')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    #ax_acc.plot(df['step'], df[metrics[0]] - df[metrics[1]], label= 'Accuracy: training minus validation')
    #ax_acc.set_ylim([0,0.1])
    # Bound terms
    ax=[]
    ylabels = ['L2 norm of Gradients_x',
               'Bound factor',
               'L2 norm of Gradients_params']
    for i in range(3):
        ax.append(fig.add_subplot(gs[i, :]))
        ax[i].plot(df['step'], df[metrics[i+2]])
        ax[i].set_yscale('log')
        ax[i].set_ylabel(ylabels[i])
    ax[0].set_title(name)
    fig.tight_layout()
    plt.show()
#plt.show()




#name,dictframe = next(iter(dataframes.items()))
def split_dictframe_epochs(dictframe, epoch_steps):
    dictframe_new = dictframe.copy()
    for key, df in dictframe_new.items():
        df['epoch'] = [int(step/epoch_steps) for step in df['step']]
    return dictframe_new

def epoch_means(df):
    #, as_index=False
    return df.groupby('epoch').agg({'value': ['mean', 'std', 'count']})


pc = 400/2409 # the pc unit relative to inchens
goldenRatio = 1.618 # ratio between width and height
marginWidth = 11.5 # width of latex margin document
textWidth   = 28#36.35
resize = 10 # scale=0.1 in latex

def margin_fig(nrows=1,ncols=1, ratio=(1,1)):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(marginWidth*pc*ratio[0], marginWidth*pc*ratio[1]/goldenRatio))
def text_fig(nrows=1,ncols=1, ratio=(1,1)):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=(textWidth*pc*ratio[0], textWidth*pc*ratio[1]/goldenRatio))
def full_fig(nrows=1,ncols=1, ratio=(1,1)):
    return plt.subplots(nrows=nrows, ncols=ncols, figsize=((textWidth+marginWidth)*pc*ratio[0], (textWidth+marginWidth*ratio[1])*pc/goldenRatio))

#fig = plt.figure()
#gs = gridspec.GridSpec(4, 4)
#ax = fig.add_subplot(gs[:, :])

fig, ax = full_fig()

metric = metrics[4]
for i, (key, dictframe) in enumerate(dataframes.items()):
    #import ipdb; ipdb.set_trace()
    name = '.'.join(key.split('\\')[-2:])
    depth = int(name.split('.')[-1][5:])
    max_step = dictframe['train/train_acc_epoch'].step.max()
    epoch_steps = max_step / 40

    dictframe = split_dictframe_epochs(dictframe, epoch_steps)

    bounds_epoch = {}
    for key, df in dictframe.items():
        if key[0] != 'b': continue
        bounds_epoch.update({key: epoch_means(df)})
        bounds_epoch[key].columns= bounds_epoch[key].columns.droplevel(0)
    #bounds_epoch[key].columns = list(map(''.join, bounds_epoch[key].columns.values))

    colour ='tab:blue' if name[:2] == 'Si' else 'tab:orange'

    ax.errorbar(depth,
                bounds_epoch[metric].iloc[0,0],
                yerr = bounds_epoch[metric].iloc[0,1] / np.sqrt(bounds_epoch[metric].iloc[0,2]),
                fmt='o',
                color=colour)

    ax.errorbar(depth,
                bounds_epoch[metric].iloc[-1,0],
                yerr = bounds_epoch[metric].iloc[-1,1] / np.sqrt(bounds_epoch[metric].iloc[-1,2]),
                fmt='s',
                color=colour)

ax.set_yscale('log')
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0],
                          marker='o', color='w', label='Simple epoch 1',
                          markerfacecolor='tab:blue', markersize=9),
                   Line2D([0], [0],
                          marker='s', color='w', label='Simple epoch 40',
                          markerfacecolor='tab:blue', markersize=9),
                   Line2D([0], [0],
                          marker='o', color='w', label='Skip epoch 1',
                          markerfacecolor='tab:orange', markersize=9),
                   Line2D([0], [0],
                          marker='s', color='w', label='Skip epoch 40',
                          markerfacecolor='tab:orange', markersize=9)]
ax.legend(handles = legend_elements, loc='upper left')
ax.set_title(metric)
ax.set_xlabel('# Hidden layers',fontsize=11)
ax.set_ylabel('Size of model dependent factor',fontsize=11)
fig.tight_layout()
plt.show()




#dictframe_new = split_dictframe_epochs(dictframe, epoch_steps)
import ipdb; ipdb.set_trace()


# TODO: As a function of depth show start and end point of bound term and x and parameters
if 0:
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 4)
    ax = fig.add_subplot(gs[:3, :])
    for key, dictframe in dataframes.items():
        name = '.'.join(key.split('\\')[-2:])
        ax.plot(dictframe[metrics[3]]['step'], dictframe[metrics[3]]['value'], label=name)
        break
    ax.set_yscale('log')
    ax.legend()
    plt.show()
# TODO: Add function to calculate epoch from steps

# TODO: Add function to extra depth from name of dict
