import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_total, y_hat_total, save_folder, epoch):
    labels = [str(round(x / 10, 1)) for x in range(37)]
    cm = confusion_matrix(y_total, y_hat_total, normalize='true', labels=labels)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, square=True, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.savefig(f'{save_folder}/confusion_matrix_{epoch}.png')
    plt.close()

def plot_training_progress(losses, acces, errs, lrs, save_folder, best_acc):
    plt.figure(figsize=(10, 10))
    
    plt.subplot(221)
    plt.plot(losses)
    plt.title(f'loss {losses[-1]}')

    plt.subplot(222)
    plt.plot(acces)
    plt.title(f'accuracy {acces[-1]:.4f}, best: {best_acc:.4f}')

    plt.subplot(223)
    plt.plot(errs)
    plt.title(f'err {errs[-1]:.4f}, best: {min(errs):.4f}')

    plt.subplot(224)
    plt.plot(lrs)
    plt.title('lr')

    plt.tight_layout()
    plt.savefig(f'{save_folder}/loss_acc.png')
    plt.close()
