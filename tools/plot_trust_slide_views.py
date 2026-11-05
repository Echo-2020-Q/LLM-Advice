"""Larger trajectory panels for projection; identical data to report Figure 6."""
from visualize_trust_scan import *
plt.rcParams.update({'font.size':14,'axes.titlesize':16})
for name,items,size in [('10_cooperation_dynamics',[('cooperation_rate','Cooperation (%)')],(16,6)),('11_trust_dynamics',[('eligible_trust_mean','Mean trust × 100'),('eligible_advice_rate','Advice use (%)')],(16,8))]:
 f,a=setup('Cooperation trajectories by simulation seed' if len(items)==1 else 'Trust and realized advice use by simulation seed','Same three diagnostic settings | seed 42: teal | seed 43: orange',len(items),3,size)
 f.subplots_adjust(left=.075,right=.96,bottom=.19,top=.78,wspace=.35,hspace=.5)
 for j,(b,p) in enumerate([(1.02,.8),(1.03,.7),(1.04,1.)]):
  ii=ids(b,p)
  for i,(col,lab) in enumerate(items):
   ax=a[i,j]
   for idx,c in zip(ii,[TEAL,ORANGE]):
    vals=H[col][idx]*100
    if col=='eligible_advice_rate':vals=pd.Series(vals).rolling(50,min_periods=1).mean().to_numpy()
    ax.plot(T,vals,color=c,lw=1.5,label=f'Seed {int(S.loc[idx,"seed"])}')
   if col=='cooperation_rate':ax.plot(T,H['control_cooperation_rate'][ii].mean(0)*100,color=GRAY,lw=1.1,label='Fermi mean')
   if i==0:ax.set_title(f'b = {b:g}, P = {p:g}')
   ax.set_ylabel(lab);ax.set_ylim(-2,103);timeaxis(ax)
   if i==len(items)-1:ax.set_xlabel('Round (log after 10)')
 a[0,0].legend(fontsize=10,loc='upper left')
 save(f,name,'Each line is one simulation run. Advice use uses a trailing 50-round average; the other series are unsmoothed.\nAdoption at round t uses previous trust; saved trust is after feedback. Two seeds do not establish outcome probabilities or a phase boundary.')
print('Created two enlarged presentation views.')
