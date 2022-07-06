#import tensorboard as tb
#experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
import pathlib
import traceback

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from experiments import export_metrics

lightning_logs =pathlib.Path("lightning_logs")
logs_models = list(lightning_logs.iterdir())
models_experiments = [path for model in logs_models for path in model.iterdir()]
#models_experiments = [models_experiments[0]]

metrics = ['train/train_acc_epoch',
           'validation/valid_acc_epoch',
           'bounds/Gradients_x',
           'bounds/Bound',
           'bounds/Gradients_x times bound',
           'bounds/Gradients parameters',
           'bounds/Gradients_x div Bound']

dataframes = {str(path) : export_metrics(str(path), metrics) for path in models_experiments}
#dataframes = next(iter(dataframes))

dictframes_all = {}
for key, dictframes in dataframes.items():
    new_key='.'.join(key.split('\\')[-2:])
    dictframes_all[new_key] = pd.concat([v.rename(columns={'value': k}) for k,v in dictframes.items()],axis=1)

# Choose one example to develop.
df = next(iter(dictframes_all.values()))
df = df.loc[:,~df.columns.duplicated()].copy()

fig = plt.figure()
gs = gridspec.GridSpec(4, 4)
# accuracy
ax_acc = fig.add_subplot(gs[3, :])
ax_acc.plot(df['step'], df[metrics[0]] - df[metrics[1]], label= 'Accuracy: training minus validation')
ax_acc.set_ylim([0,0.1])
# Bound terms
ax=[]
for i in range(3):
    ax.append(fig.add_subplot(gs[i, :]))
    ax[i].plot(df['step'], df[metrics[i+2]])
    ax[i].set_yscale('log')
fig.tight_layout()
#plt.show()
# TODO: As a function of depth show start and end point of bound term and x and parameters
