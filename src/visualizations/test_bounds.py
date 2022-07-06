import os
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

from experiments import export_metrics

# TODO: Get list of folders in a folder path to navigate through to export data from tb
lightning_logs =pathlib.Path("lightning_logs")
logs_models = list(lightning_logs.iterdir())
#models_experiments = {model.name: [path for path in model.iterdir()] for model in logs_models}
models_experiments = [path for model in logs_models for path in model.iterdir()]
# TODO: Define metrics to be used from each experiment

metrics = ['bounds/Gradients_x times bound',
           'bounds/Gradients parameters']
#df_test = export_metrics(str(models_experiments['SimpleMLP'][0]), metrics)
# TODO: How to loop through. Either change models_experiments or change the first above.
#models_experiments['SimpleMLP'][0]
dataframes = {str(path) : export_metrics(str(path), metrics) for path in models_experiments}

dictframes_all = {}
for key, dictframes in dataframes.items():
    new_key='.'.join(key.split('\\')[-2:])
    dictframes_all[new_key] = pd.concat([v.rename(columns={'value': k}) for k,v in dictframes.items()],axis=1)
    # k..split('/')[-1]

c=0
for k,df in dictframes_all.items():
    c += (df[metrics[1]] < df[metrics[0]]).any()

print(c)

#dataframes = {model.name+path.name: export_metrics(str(path), metrics) for model, path in model.iterdir()] for model in logs_models}

# TODO: prep data
def metric_to_legend(metric): return metric.split('/')[-1]


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





# TODO: Make nice print ready figure
df = next(iter(dictframes_all.values()))
df = df.loc[:,~df.columns.duplicated()].copy()

fig, ax = text_fig()

#xb,= ax.plot(df['step'],df['Gradients_x times bound'], label = 'Grad_x * factor')
#params,= ax.plot(df['step'],df['Gradients parameters'], label = 'Grad_param')
difference, = ax.plot(df['step'],df[metrics[1]] / df[metrics[0]], label = 'Grad_param / (Grad_x * factor)')
ax.set_yscale("log")

#ax.plot(epoch_to_load*np.ones((5)),np.linspace(0.00,0.01,5), 'k:')
#ax.set_xlim([0.,15])
#ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
#ax.set_ylim([0.,0.01])
ax.set_xlabel('Step',fontsize=11)
ax.set_ylabel('L2 Norm',fontsize=11)
ax.legend()#loc='lower left')

#ax_acc = ax.twinx()
#va, = ax_acc.plot(np.arange(1,valid_loss.shape[0]+1),valid_acc, label='Validation accuracy', c='tab:green')
#ax_acc.set_ylabel('Accuracy',fontsize=11)
#ax_acc.yaxis.label.set_color('tab:green')
#ax_acc.set_ylim([0.9,1.0])

#ax_acc.legend(loc='upper left')
#ax.legend([va,tl,vl],['Validation accuracy','Train loss','Valid loss'], loc='center right')
#ax.legend([tl,vl],['Training loss','Validation loss'])
ax.grid(True)
#ax.yaxis.set_major_formatter(FuncFormatter(formatFloat))#formatStrFormatter('%.3f'.lstrip('0')))
fig.tight_layout()
plt.show()
