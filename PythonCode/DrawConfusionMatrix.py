import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path

data_path = Path('D:\Data\Lobster\Lobster_Recording-200319-161008\EventClassifierDataset\output_IL.mat')

data = loadmat(str(data_path.absolute()))
mat_confusion = data.get('mat_confusion')

# Draw End result
#cmap = sns.color_palette(['#ffffff','#ff0000'], n_colors=50, as_cmap=True)
#cmap = sns.light_palette('#ff0000', as_cmap=True)
cmap = sns.light_palette('#0000C0', as_cmap=True)
f = plt.figure(2, figsize=(12, 6))
f.clear()
ax1, ax2 = f.subplots(1,2)
f.suptitle('IL',fontsize = 25)
confusion_mat_shuffled = mat_confusion[:,0:3,:]
confusion_mat_real = mat_confusion[:,3:,:]
num_sample = np.sum(np.sum(confusion_mat_shuffled,axis=2),axis=1).reshape(-1,1)

sns.heatmap(np.sum(confusion_mat_shuffled,axis=2) / num_sample, ax=ax1, cmap=cmap, vmin=0, vmax=1, annot=True, annot_kws={'fontsize':15,'color':'k'}, square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.heatmap(np.sum(confusion_mat_real,axis=2) / num_sample, ax=ax2, cmap=cmap, vmin=0, vmax=1, annot=True, annot_kws={'fontsize':15,'color':'k'}, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# present std
confusion_mat_std_shuffled = np.std(confusion_mat_shuffled,axis=2) / num_sample
confusion_mat_std_real = np.std(confusion_mat_real,axis=2) / num_sample
for i in np.arange(3):
    for j in np.arange(3):
        ax1.text(0.5 + i, 0.6 + j,'(±{:.3f})'.format(confusion_mat_std_shuffled[i,j]),verticalalignment='top',horizontalalignment='center',color='w')
        ax2.text(0.5 + i, 0.6 + j,'(±{:.3f})'.format(confusion_mat_std_real[i,j]), verticalalignment = 'top', horizontalalignment = 'center', color = 'w')

Y_label = ['Head Entry', 'Avoidance HW', 'Escape HW']
ax1.set_title('shuffled',fontsize=18)
ax1.set_xticklabels(Y_label,verticalalignment='center')
ax1.set_yticklabels(Y_label,verticalalignment='center')
ax1.set_xlabel('predicted',fontsize=15)
ax1.set_ylabel('actual',labelpad=10,fontsize=15)
ax2.set_title('real',fontsize=19)
ax2.set_xticklabels(Y_label,verticalalignment='center')
ax2.set_yticklabels(Y_label,verticalalignment='center')
ax2.set_xlabel('predicted',fontsize=15)
ax2.set_ylabel('actual',labelpad=10,fontsize=15)
plt.show()

