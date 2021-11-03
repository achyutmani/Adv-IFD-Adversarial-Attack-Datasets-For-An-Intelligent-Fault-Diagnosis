import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt

data=np.array([[100,87.85,85.45],[87.85,100,99.18],[87.87,87.95,100]])
#data=np.array([[],[],[]])
data[np.diag_indices_from(data)]=100	
cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=False)
heat_map = sb.heatmap(data,annot=True, fmt=".2f",vmin=0, vmax=100,annot_kws={"size": 50},cmap=cmap,linewidths=5)
Size=50
heat_map.set_xticklabels(['M1','M2','M3'],size=Size)
heat_map.set_yticklabels(['M1','M2','M3'],size=Size)
plt.ylabel('Source Model',size=Size)
plt.xlabel('Target Model',size=Size)
figure = plt.gcf()  # get current figure
figure.set_size_inches(20,10) # set figure's size manually to your full screen (32x18)
plt.savefig('BIM_HM_CWRU.pdf', bbox_inches='tight')
plt.show()
data[np.diag_indices_from(data)]=0
print("Average Fooling Rate=\n",np.mean(data,axis=0))
