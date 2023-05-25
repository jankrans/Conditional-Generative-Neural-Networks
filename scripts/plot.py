import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plotday(data, labels=None, title=None, zeroline=False, colors=['orange', 'steelblue','teal']):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    #create pandaseries to get the correct ticks
    idx = pd.date_range('00:00', '23:59', freq='15min')
    testSeries = pd.Series(data=np.random.randn(len(idx)), index=idx)
    ticks = testSeries.index[testSeries.index.minute == 0]

    if np.ndim(data) == 1: data = [data]
    for i, d in enumerate(data):
        series = pd.Series(d, index=idx)
        if labels is None: 
            graph_col = 'orange' if i == len(data)-1 else 'black'
            graph_alpha = 1 if i == len(data)-1 else 0.2
            series.plot(ax=ax, color=graph_col,alpha=graph_alpha)
        else:
            series.plot(ax=ax, label=labels[i], color=colors[i],alpha=0.6)
    
    ax.legend()
    ax.set_title(title if title else 'Daily consumtpion (kwH)')
    ax.set_ylabel('Consumption (kwH)')
    
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.strftime('%H:%M'))
    n=2
    # [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    ax.grid(True, axis='x')
    if zeroline: ax.axhline(y = 0, color = 'red')
    
    plt.show()

def plot_losses(train_losses, val_losses, test_losses=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    if test_losses: plt.plot(epochs, test_losses, 'g', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()