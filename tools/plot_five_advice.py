#!/usr/bin/env python3
"""English annotated figures for five observed replies per state and model."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
ROOT=Path('/mnt/d/pycharm_community_edition_2024_01_04/py_projects/llm_advice')
if not ROOT.is_dir():ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/1101_3LLM_Analysis';DATA=OUT/'data';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
MODELS=['GPT','DeepSeek','Qwen'];COLORS={'GPT':'#4776af','DeepSeek':'#cb8138','Qwen':'#138b80'}
NAVY='#17394f';GRAY='#566e7e'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'axes.titlesize':15,'axes.titleweight':'bold','text.color':NAVY,'axes.labelcolor':NAVY,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b8c7d1','xtick.labelsize':11.5,'ytick.labelsize':11.5,'grid.color':'#dde6eb','savefig.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'none'})
S=pd.read_csv(DATA/'model_summary.csv').set_index('model').loc[MODELS]
A=pd.read_csv(DATA/'aligned_states.csv');R=pd.read_csv(OUT/'merged/combined_answers.csv')
assert len(R)==7680 and len(A)==512 and R.groupby(['model_label','grid_key']).size().eq(5).all()
AUDITS=[]
def read(name):return pd.read_csv(DATA/f'{name}.csv')
def setup(title,subtitle,rows=1,cols=3,size=(13.2,5.8),top=.75,bottom=.23):
 f,a=plt.subplots(rows,cols,figsize=size,squeeze=False);f.suptitle(title,x=.065,y=.98,ha='left',fontsize=20,fontweight='bold');f.text(.065,.905,subtitle,fontsize=11.5,color=GRAY,va='top');f.subplots_adjust(left=.08,right=.97,top=top,bottom=bottom,wspace=.36,hspace=.55);return f,a

def save(f,name,note):
 f.text(.065,.035,note,fontsize=10,color=GRAY,va='bottom',linespacing=1.45)
 f.canvas.draw();rend=f.canvas.get_renderer();bad=[]
 # Check drawn figure text and labels; do not count unused automatic ticks outside axis limits.
 from matplotlib.text import Text
 for t in f.findobj(Text):
  if not t.get_visible() or not t.get_text().strip():continue
  if t in [q for ax in f.axes for axis in (ax.xaxis,ax.yaxis) for tick in axis.get_major_ticks() for q in (tick.label1,tick.label2)]:continue
  b=t.get_window_extent(rend)
  if b.x0<-.5 or b.y0<-.5 or b.x1>f.bbox.width+.5 or b.y1>f.bbox.height+.5:bad.append(t.get_text())
 assert not bad,(name,bad)
 for ext in ['png','pdf','svg']:f.savefig(FIG/f'{name}.{ext}',dpi=230)
 AUDITS.append({'figure':name,'text_within_canvas':True});plt.close(f)

def overview():
 f,a=setup('Three models differ in advice, persistence and repeat agreement','All 512 local states receive equal weight; each model contributes 2,560 recorded replies.')
 specs=[('p_C','(a) Advice to cooperate','Replies recommending C (%)'),('current_action_retention','(b) Retain the current action','Replies retaining current C or D (%)'),('within_state_pair_agreement','(c) Pairwise agreement','Same-action pairs within a state (%)')]
 for ax,(col,title,ylab) in zip(a[0],specs):
  vals=S[col].to_numpy()*100;ax.bar(MODELS,vals,color=[COLORS[m] for m in MODELS],width=.65)
  for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.2f}%',ha='center',fontsize=13,fontweight='bold')
  ax.set(title=title,ylabel=ylab,ylim=(0,106),yticks=[0,25,50,75,100]);ax.grid(axis='y',alpha=.55);ax.set_axisbelow(True)
 save(f,'01_overview','X: the model label recorded in the logs. (a,b) Denominator = 2,560 replies per model.\n(c) For each state, count agreeing pairs among the 10 distinct pairs of its five replies; then average across 512 states.\nAgreement measures consistency of recorded choices, not correctness or expected game payoff.')

def transitions():
 f,a=setup('Preserving cooperation differs from converting a defector','Each row conditions on the focal player’s current action; C = cooperate, D = defect.',size=(13.2,5.6))
 for ax,m in zip(a[0],MODELS):
  vals=np.array([[S.loc[m,'p_C_given_current_C'],1-S.loc[m,'p_C_given_current_C']],[S.loc[m,'p_C_given_current_D'],1-S.loc[m,'p_C_given_current_D']]])
  im=ax.imshow(vals,vmin=0,vmax=1,cmap='YlGnBu');ax.set(xticks=[0,1],xticklabels=['Recommend C','Recommend D'],yticks=[0,1],yticklabels=['C','D'],title=m,xlabel='Recommended next action',ylabel='Current focal action')
  for i in range(2):
   for j in range(2):ax.text(j,i,f'{vals[i,j]*100:.2f}%\n{round(vals[i,j]*1280):,} / 1,280',ha='center',va='center',fontsize=13,color='white' if vals[i,j]>.62 else NAVY)
 save(f,'02_transitions','Rows: 256 exact states with the same focal action × 5 replies = 1,280 replies per model; row percentages sum to 100%.\nCell shading and labels show the observed recommendation frequency. These are advice choices, not realized strategy transitions.')

def neighbors():
 N=read('neighbor_response');f,a=setup('Qwen more often recommends C to a surrounded defector','Direct neighbors are north, south, east and west; the four corners are pooled within each group.',cols=2,size=(13.2,5.8))
 for ax,center in zip(a[0],['C','D']):
  for m in MODELS:
   d=N[(N.model==m)&(N.center==center)].sort_values('cardinal_c');ax.plot(d.cardinal_c,100*d.p_C,marker='o',lw=2.2,color=COLORS[m],label=m)
  ax.set(title=f'Focal player currently {center}',xlabel='Number of cooperating direct neighbors',ylabel='Replies recommending C (%)',xticks=range(5),ylim=(-3,104),yticks=[0,25,50,75,100]);ax.grid(alpha=.5)
 a[0,0].legend(fontsize=11)
 save(f,'03_neighbors','X: 0–4 cooperating direct neighbors. Y: C replies / all replies in that focal-action and neighbor-count group.\nPer model and focal action, denominators at x = 0, 1, 2, 3, 4 are 80, 320, 480, 320, 80 replies respectively.\nLines connect descriptive group averages; no simulation outcome or confidence interval is shown.')

def context():
 C=read('context_response');f,a=setup('Both direct and diagonal context distinguish recorded advice','Each cell pools all arrangements with the indicated two counts; focal C and D are shown separately.',rows=2,cols=3,size=(13.2,8.4),top=.81,bottom=.19)
 f.subplots_adjust(right=.91,wspace=.36,hspace=.55)
 for i,center in enumerate(['C','D']):
  for j,m in enumerate(MODELS):
   ax=a[i,j];d=C[(C.model==m)&(C.center==center)];v=d.pivot(index='diagonal_c',columns='cardinal_c',values='p_C').reindex(index=range(5),columns=range(5)).to_numpy()*100
   im=ax.imshow(v,origin='lower',vmin=0,vmax=100,cmap='YlGnBu',aspect='auto');ax.set(title=f'{m} | focal {center}',xticks=range(5),yticks=range(5),xlabel='Cooperating direct neighbors',ylabel='Cooperating diagonal neighbors')
   for r in range(5):
    for c in range(5):ax.text(c,r,f'{v[r,c]:.0f}',ha='center',va='center',fontsize=12,color='white' if v[r,c]>62 else NAVY)
 cax=f.add_axes([.93,.25,.015,.50]);f.colorbar(im,cax=cax,label='Replies recommending C (%)')
 save(f,'04_context','Numbers and color = C replies as a percentage of all replies in that cell, rounded to whole percent.\nX counts four direct neighbors; Y counts four diagonal neighbors. Diagonal neighbors appear in the prompt but do not directly pay the focal player.\nFor x=k and y=d, denominator = 5 × choose(4,k) × choose(4,d), ranging from 5 to 180 replies per model and focal action.')

def consistency():
 V=read('vote_histogram');f,a=setup('Five replies expose disagreement hidden by a single answer','For each exact state, k counts how many of its five replies recommend C.',size=(13.2,6.1))
 for ax,m in zip(a[0],MODELS):
  d=V[V.model.eq(m)].sort_values('C_answers_out_of_5');vals=d.fraction_of_512_states.to_numpy()*100;ax.bar(range(6),vals,color=COLORS[m],alpha=.9)
  for k,(v,n) in enumerate(zip(vals,d.n_states)):ax.text(k,v+1,f'{n}',ha='center',fontsize=11)
  ax.set(title=f'{m}\nUnanimous: {int(S.loc[m,"unanimous_states"])}/512 ({S.loc[m,"unanimity_fraction"]*100:.1f}%)',xlabel='Number of C replies out of five (k)',ylabel='Share of 512 exact states (%)',xticks=range(6),yticks=[0,20,40,60],ylim=(0,66));ax.grid(axis='y',alpha=.5);ax.set_axisbelow(True)
 save(f,'05_consistency','X: k=0 means five D replies; k=5 means five C replies. Bars k=1…4 contain mixed replies. Labels above bars are state counts.\nY denominator = 512 exact states, not 2,560 replies. Unanimity combines k=0 and k=5.\nFive identical observed answers do not establish a true response probability of 0 or 1; repeat pairs share replies and are not independent trials.')

def model_agreement():
 f,a=setup('Model differences depend on the exact local state','Each cell counts states with the indicated pair of observed C frequencies.',size=(13.2,6.4),bottom=.25)
 pairs=[('GPT','DeepSeek'),('GPT','Qwen'),('DeepSeek','Qwen')]; matrices=[]
 for x,y in pairs:
  z=np.zeros((6,6),dtype=int)
  for k,l in zip(A[f'{x}_C'],A[f'{y}_C']):z[int(l),int(k)]+=1
  assert z.sum()==512;matrices.append(z)
 vmax=max(z.max() for z in matrices)
 f.subplots_adjust(right=.90,wspace=.43)
 for ax,(x,y),z in zip(a[0],pairs,matrices):
  im=ax.imshow(z,origin='lower',vmin=0,vmax=vmax,cmap='Blues',aspect='equal');ax.plot([-.5,5.5],[-.5,5.5],color='#8e9ba4',ls='--',lw=.8)
  dis=((A[f'{x}_C']>=3)!=(A[f'{y}_C']>=3)).sum()
  ax.set(title=f'{x} vs {y}\nMajority differs: {dis}/512',xticks=range(6),xticklabels=[0,20,40,60,80,100],yticks=range(6),yticklabels=[0,20,40,60,80,100],xlabel=f'{x}: C replies / 5 (%)',ylabel=f'{y}: C replies / 5 (%)')
  for r in range(6):
   for c in range(6):
    if z[r,c]:ax.text(c,r,str(z[r,c]),ha='center',va='center',fontsize=11.5,color='white' if z[r,c]>vmax*.55 else NAVY)
 cax=f.add_axes([.92,.30,.014,.40]);f.colorbar(im,cax=cax,label='Number of exact states')
 save(f,'06_model_agreement','X and Y: empirical C frequency within five replies to the SAME state, taking values 0%, 20%, …, 100%.\nNumbers/color = count of states, totaling 512 in each panel; blank cells have zero states. Dashed diagonal = equal C frequencies.\nMajority means at least 3 C replies out of 5; majority disagreement differs from a difference in empirical frequencies.')

def batches():
 B=read('batch_summary');f,a=setup('Aggregate advice frequencies are similar across the two batches','Old and new batch records remain distinguishable after merging; matching overall rates does not prove identical behavior.',size=(13.2,6.1),top=.72,bottom=.25)
 for ax,col,title in zip(a[0],['p_C','p_C_given_current_C','p_C_given_current_D'],['All focal actions','Focal player currently C','Focal player currently D']):
  for x,m in enumerate(MODELS):
   d=B[B.model.eq(m)].set_index('batch');v1=d.loc['old',col]*100;v2=d.loc['new',col]*100
   ax.plot([x-.13,x+.13],[v1,v2],color=COLORS[m],lw=2)
   ax.scatter(x-.13,v1,facecolors='white',edgecolors=COLORS[m],s=75,lw=2,label='Old batch' if x==0 else None,zorder=4)
   ax.scatter(x+.13,v2,color=COLORS[m],s=75,label='New batch' if x==0 else None,zorder=4)
   ax.text(x,max(v1,v2)+7,f'{v2-v1:+.2f} pp',ha='center',fontsize=11)
  ax.set(title=title,xticks=range(3),xticklabels=MODELS,xlim=(-.5,2.5),ylim=(-3,106),yticks=[0,25,50,75,100],ylabel='Replies recommending C (%)');ax.grid(axis='y',alpha=.5)
 a[0,0].legend(loc='upper left',fontsize=10)
 save(f,'07_batch_comparison','Hollow marker = old batch; filled marker = new batch. Annotations = new minus old, in percentage points (pp).\nEach batch covers the same 512 states (256 per focal action). Replies per state: GPT/Qwen 2 old + 3 new; DeepSeek 1 old + 4 new.\nAll retained outputs are below 10,240 tokens; no token-limit truncation was observed. Pooling weights individual replies, not batches equally.')

def examples():
 cases=[('CCCCCCCCC','All C'),('DDDDDDDDD','All D'),('CCCCDCCCC','D surrounded by C'),('CDCDCDCDC','C with four direct D')]
 f=plt.figure(figsize=(13.2,8.5));f.suptitle('Exact states make agreement and disagreement concrete',x=.065,y=.98,ha='left',fontsize=20,fontweight='bold');f.text(.065,.918,'The full 3 × 3 state is shown; the orange border marks the focal center.',fontsize=12,color=GRAY)
 grid=f.add_gridspec(2,2,left=.07,right=.97,top=.86,bottom=.19,wspace=.19,hspace=.36)
 for z,(key,title) in enumerate(cases):
  ax=f.add_subplot(grid[z//2,z%2]);ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,5.2);ax.set_title(title,loc='left',fontsize=15)
  for i,ch in enumerate(key):
   row,col=divmod(i,3);x=.1+col*.83;y=3.9-row*.83
   ax.add_patch(Rectangle((x,y),.79,.79,facecolor='#15978b' if ch=='C' else '#e7edf1',edgecolor='white'))
   ax.text(x+.395,y+.395,ch,ha='center',va='center',color='white' if ch=='C' else NAVY,fontsize=15,fontweight='bold')
  ax.add_patch(Rectangle((.93,3.07),.79,.79,fill=False,lw=2.5,edgecolor='#e5a342'))
  ax.text(3.35,4.65,'Recorded replies: old | new',fontsize=11,color=GRAY)
  for i,m in enumerate(MODELS):
   d=R[(R.model_label==m)&(R.grid_key==key)].sort_values('repetition');old=' '.join(d.loc[d.batch_label=='old','recommendation']);new=' '.join(d.loc[d.batch_label=='new','recommendation']);n=int(d.is_c.sum())
   ax.text(3.25,3.76-i*.93,m,fontsize=13,fontweight='bold',color=COLORS[m]);ax.text(5.75,3.76-i*.93,f'{old} | {new}',fontsize=13,fontfamily='DejaVu Sans Mono');ax.text(9.75,3.76-i*.93,f'{n}/5 C',ha='right',fontsize=12)
  ax.text(.1,.40,' / '.join([key[i:i+3] for i in [0,3,6]]),fontsize=11,color=GRAY)
 save(f,'08_examples','Each letter is one retained recommendation, shown in merged repetition order; the vertical bar separates collection batches.\nThe final column counts C recommendations out of five. These four examples are illustrative, not a representative sample of all 512 states.\nC/D labels refer to actions, not better/worse outcomes; the observed replies do not establish optimal decisions.')

def center_effect():
 E=read('center_pair_effects');Q=read('center_pair_summary').set_index('model');f,a=setup('DeepSeek most strongly favors C when the focal action is C','Match each state to the state with identical eight surrounding cells and the opposite focal action.',size=(13.2,6.3),bottom=.25)
 xs=np.arange(-5,6)*20
 for ax,m in zip(a[0],MODELS):
  v=np.rint(E.loc[E.model==m,'center_C_minus_D_difference']*5).astype(int).value_counts();counts=np.array([v.get(k,0) for k in range(-5,6)])
  assert counts.sum()==256
  ax.bar(xs,counts/256*100,width=16,color=COLORS[m]);ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set(title=f'{m}\nMean contrast = {Q.loc[m,"mean_center_C_minus_D_difference"]*100:+.2f} pp',xlabel='C-frequency contrast (percentage points)',ylabel='Share of 256 matched state pairs (%)',xticks=[-100,-50,0,50,100],xlim=(-113,113),ylim=(0,74),yticks=[0,20,40,60]);ax.grid(axis='y',alpha=.5);ax.set_axisbelow(True)
 save(f,'09_center_effect','X = 100 × [C replies/5 with center C − C replies/5 with center D], holding all eight neighbors fixed.\nY = number of matched pairs at that contrast / 256. Positive values mean more C recommendations when the center is already C.\nChanging the center also changes its stated payoff where direct C neighbors exist; this is not an isolated causal test of action persistence.')

if __name__=='__main__':
 overview();transitions();neighbors();context();consistency();model_agreement();batches();examples();center_effect()
 (OUT/'review').mkdir(exist_ok=True)
 (OUT/'review/figure_validation.json').write_text(json.dumps(AUDITS,indent=2))
 print('Generated nine annotated figures in PNG / PDF / SVG.')
