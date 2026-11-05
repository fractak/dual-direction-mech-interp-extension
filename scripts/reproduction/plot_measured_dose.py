import json
from pathlib import Path
import matplotlib.pyplot as plt

out = Path('artifacts/reproduced')
conditions=json.load(open('artifacts/reproduced/results/pilot_results.json'))['conditions']
fig,axes=plt.subplots(1,2,figsize=(9.4,3),layout='constrained',sharey=True)
for ax,metric,title in zip(axes,['acknowledgment','unsupported'],['Acknowledgment','Unsupported assertions']):
    for receiver,keys,color,label,offset in [
        ('original_instruct',['c0000','c0008','c0009','c0010'],'#3274a1','Original Instruct',-.55),
        ('safety_lora',['c0029','c0037','c0038','c0039'],'#bc682e','Adapted',.55)]:
        ax.plot([0,3,10,30],[conditions[k]['counts'][metric] for k in keys],'-o',color=color,label=label)
        for budget in [3,10,30]:
            values=[c['counts'][metric] for c in conditions.values() if c['receiver']==receiver and c['source']=='random' and c['amount']==budget/100]
            assert len(values)==6
            ax.vlines(budget+offset,min(values),max(values),color=color,alpha=.7)
            ax.scatter([budget+offset]*6,values,marker='_',color=color,s=35,alpha=.8)
    ax.set(title=title,xlabel='Relative displacement budget (%)',xticks=[0,3,10,30],ylim=(-3,63),xlim=(-1.5,32))
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Responses (of 60)')
axes[1].legend(loc='upper left',frameon=False)
fig.savefig(out/'figures/measured_relative_dose.pdf')
plt.close(fig)
