#!/usr/bin/env python3
"""Plot audited adaptive trust data without running simulation/model APIs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import json, hashlib
ROOT=Path(__file__).resolve().parents[1];LOWER=Path(str(ROOT).lower())
if LOWER.exists() and LOWER.samefile(ROOT):ROOT=LOWER
OUT=ROOT/'reports/Trust_1030';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
SRC=ROOT/'main/SPD/results/scan_20261029_183749_initial'
S=pd.read_csv(OUT/'data/validated_runs.csv').sort_values(['b','llm_access_fraction','seed']).reset_index(drop=True)
META=json.loads((OUT/'data/cache_metadata.json').read_text())
for artifact,digest in META['artifact_sha256'].items():
 assert hashlib.sha256((OUT/'data'/artifact).read_bytes()).hexdigest()==digest,artifact
H=np.load(OUT/'data/trajectory_cache.npz')
ALL_OLD=pd.read_csv(ROOT/'main/SPD/results/scan_20261027_121246_2次gpt微扫描/summary.csv')
OLD=ALL_OLD[ALL_OLD.lambda_prob.eq(.5)].sort_values(['b','llm_access_fraction','seed']).reset_index(drop=True)
assert np.array_equal(S[['b','llm_access_fraction','seed']],OLD[['b','llm_access_fraction','seed']])
assert H['cooperation_rate'].shape==(154,10001)
assert np.allclose(H['cooperation_rate'][:,-1000:].mean(1),S.tail_mean_cooperation_rate)
BS=sorted(S.b.unique());PS=sorted(S.llm_access_fraction.unique());T=np.arange(10001)
NAVY='#18394e';TEAL='#008c86';ORANGE='#d57735';PURPLE='#8c6aac';GRAY='#788896'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#b4c3cd','grid.color':'#e0e8eb','savefig.facecolor':'white'})
def setup(title,sub,rows=1,cols=2,size=(15,7)):
 f,ax=plt.subplots(rows,cols,figsize=size,squeeze=False);f.suptitle(title,x=.06,y=.975,ha='left',fontsize=21,fontweight='bold');f.text(.06,.913,sub,fontsize=11,color=GRAY,va='top');f.subplots_adjust(left=.065,right=.955,bottom=.16,top=.80,wspace=.32,hspace=.5);return f,ax
def save(f,name,note):
 f.text(.06,.035,note,fontsize=9,color=GRAY,va='bottom')
 f.canvas.draw()
 from matplotlib.text import Text
 bounds=f.bbox; renderer=f.canvas.get_renderer(); bad=[]
 for artist in f.findobj(Text):
  if artist.get_visible() and artist.get_text().strip():
   box=artist.get_window_extent(renderer)
   if box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.width+1 or box.y1 > bounds.height+1:bad.append(artist.get_text())
 if bad:raise ValueError(f'Figure text outside canvas {name}: {bad}')
 for ext in ['png','pdf','svg']:f.savefig(FIG/f'{name}.{ext}',dpi=220)
 plt.close(f)
def matrix(col,frame=S):return frame.groupby(['b','llm_access_fraction'])[col].mean().unstack().reindex(index=BS,columns=PS).to_numpy()
def heat(f,ax,data,title,label='%',dec=0):
 cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#edf0f2');im=ax.imshow(data,vmin=0,vmax=100,cmap=cmap,aspect='auto')
 ax.set(xticks=range(11),xticklabels=[f'{p:g}' for p in PS],yticks=range(7),yticklabels=[f'{b:g}' for b in BS],xlabel='Advice eligibility P',ylabel='Temptation b',title=title)
 for i in range(7):
  for j in range(11):
   v=data[i,j];s='N/A' if np.isnan(v) else f'{v:.{dec}f}';c=GRAY if np.isnan(v) else ('white' if v>62 else NAVY);ax.text(j,i,s,ha='center',va='center',fontsize=8.1,color=c)
 cb=f.colorbar(im,ax=ax,fraction=.042,pad=.025);cb.set_label(label)
def ids(b,p):return S.index[np.isclose(S.b,b)&np.isclose(S.llm_access_fraction,p)].to_numpy()
def timeaxis(ax):
 ax.set_xscale('symlog',linthresh=10);ax.set_xlim(0,10000);ax.set_xticks([0,10,100,1000,10000],['0','10','100','1k','10k']);ax.get_xticklabels()[-1].set_ha('right');ax.grid(alpha=.65)
def main():
 f,a=setup('Adaptive trust expands cooperation at low temptation','GPT advice table | 77 parameter settings × 2 seeds | final window: rounds 9001–10000')
 heat(f,a[0,0],100*matrix('tail_mean_cooperation_rate'),'(a) Adaptive: mean cooperation',label='Cooperation (%)')
 heat(f,a[0,1],100*matrix('delta_tail_mean_cooperation_rate'),'(b) Adaptive minus paired Fermi',label='Difference (percentage points)')
 save(f,'01_cooperation','Each cell is a two-seed mean. b values are discrete scan settings, not equally spaced numeric intervals.\nAt b = 1.04, P = 0.9 and 1, means mix an extinct run with a surviving run; see the seed-specific trajectories.')
 f,a=setup('Compare with the same GPT table at fixed adoption λ = 0.5','Same recorded lattice, horizon, seeds and advice-source hashes; historical comparison remains descriptive.',1,3,(17,7))
 for j,(ax,data,title) in enumerate(zip(a[0],[100*matrix('tail_mean_cooperation_rate',OLD),100*matrix('tail_mean_cooperation_rate'),100*(matrix('tail_mean_cooperation_rate')-matrix('tail_mean_cooperation_rate',OLD))],['(a) Fixed λ = 0.5','(b) Adaptive, initial trust = 0.5','(c) Adaptive minus fixed'])):heat(f,ax,data,title,label='Percentage points' if j==2 else 'Cooperation (%)')
 save(f,'02_fixed_comparison','Active settings (P > 0): adaptive is higher in 23/70 settings, tied in 47/70, lower in 0/70, based on two-seed means.\nFixed λ = 1 can outperform adaptive trust in some settings (Figure 3). Changes in adoption intensity and heterogeneity are not separated.')
 f,a=setup('Coverage response depends on the comparison policy','Final-window cooperation; individual seed dots and shaded observed ranges. Fixed λ = 1 is also shown.',2,2,(15,9))
 for ax,b in zip(a.flat,BS[:4]):
  for frame,col,label,c in [(S,'tail_mean_cooperation_rate','Adaptive',TEAL),(OLD,'tail_mean_cooperation_rate','Fixed λ = 0.5',ORANGE),(ALL_OLD[ALL_OLD.lambda_prob.eq(1)],'tail_mean_cooperation_rate','Fixed λ = 1',PURPLE),(S,'control_tail_mean_cooperation_rate','Fermi control',GRAY)]:
   g=frame[frame.b.eq(b)].groupby('llm_access_fraction')[col].agg(['mean','min','max']);ax.plot(g.index,100*g['mean'],color=c,lw=1.8,label=label);ax.fill_between(g.index,100*g['min'],100*g['max'],color=c,alpha=.10)
   if label!='Fermi control':ax.scatter(frame.loc[frame.b.eq(b),'llm_access_fraction'],100*frame.loc[frame.b.eq(b),col],color=c,s=10,alpha=.65)
  ax.set(title=f'b = {b:g}',xlabel='Advice eligibility P',ylabel='Cooperation (%)',ylim=(-2,103),xticks=np.arange(0,1.1,.2));ax.grid(alpha=.6)
 a[0,0].legend(loc='upper left',fontsize=8.5,ncol=2)
 save(f,'03_coverage','Bands are two-run ranges, not confidence intervals. At b = 1.02, P = 0.6: adaptive 50.49%, fixed λ = 1 89.72%.\nAdaptive is below the best tested fixed policy in 27/70 active cells; choosing that policy after observing results is exploratory.')
 f,a=setup('Trust becomes high with cooperation and low after extinction','Trust statistics include eligible positions only; P = 0 has no eligible individuals.')
 heat(f,a[0,0],100*matrix('tail_mean_eligible_trust'),'(a) Final-window mean trust',label='Trust × 100',dec=1)
 heat(f,a[0,1],100*matrix('final_eligible_low_trust_fraction'),'(b) Final share with trust < 0.1',label='Eligible individuals (%)')
 save(f,'04_trust','Trust is a probability of adopting advice, not a calibrated measure of advice accuracy. Actual adoption closely follows mean trust.\nNear-unit trust can include continued reinforcement from tiny aspiration gaps after payoff becomes constant (Figure 9).')
 f,a=setup('Trajectories separate recovery, coexistence and seed dependence','Each seed is shown separately. Advice use is the fraction of eligible individuals adopting in a round.',3,3,(16,11))
 for j,(b,p) in enumerate([(1.02,.8),(1.03,.7),(1.04,1.)]):
  ii=ids(b,p)
  for i,(col,lab) in enumerate([('cooperation_rate','Cooperation (%)'),('eligible_trust_mean','Mean trust × 100'),('eligible_advice_rate','Advice use (%)')]):
   ax=a[i,j]
   for idx,c in zip(ii,[TEAL,ORANGE]):
    vals=H[col][idx]*100
    if col=='eligible_advice_rate':vals=pd.Series(vals).rolling(50,min_periods=1).mean().to_numpy()
    ax.plot(T,vals,color=c,lw=1.15,label=f'Seed {int(S.loc[idx,"seed"])}')
   if i==0:ax.plot(T,H['control_cooperation_rate'][ii].mean(0)*100,color=GRAY,lw=1,label='Fermi mean');ax.set_title(f'b = {b:g}, P = {p:g}')
   ax.set_ylabel(lab);ax.set_ylim(-2,103);timeaxis(ax)
   if i==2:ax.set_xlabel('Round (linear to 10, log thereafter)')
 a[0,0].legend(fontsize=9)
 save(f,'05_dynamics','Advice-use curves use a trailing 50-round average. History row t records post-feedback trust; adoption in that row uses trust from t−1.\nTrust and cooperation are jointly determined; the observed correlation does not establish which causes which.')
 f,a=setup('Individual trust distributions differ across regimes and seeds','Saved ten-bin distributions among eligible individuals at round 10,000; individual trust maps were not saved.',1,3,(16,6.6))
 for ax,(b,p) in zip(a[0],[(1.02,.1),(1.03,.7),(1.04,1.)]):
  for idx,c,offset in zip(ids(b,p),[TEAL,ORANGE],[-.19,.19]):ax.bar(np.arange(10)+offset,[S.loc[idx,f'final_eligible_trust_bin_{k}']*100 for k in range(10)],width=.38,color=c,alpha=.85,label=f'Seed {int(S.loc[idx,"seed"])}')
  ax.set(title=f'b = {b:g}, P = {p:g}',xticks=range(10),xticklabels=[f'{v/10:.1f}' for v in range(10)],xlabel='Trust-bin lower edge',ylabel='Eligible individuals (%)',ylim=(0,100));ax.grid(axis='y',alpha=.5)
 a[0,0].legend(fontsize=9)
 save(f,'06_trust_distribution','Bins are [0, 0.1), …, [0.8, 0.9), [0.9, 1]. The low-trust statistic uses the strict threshold trust < 0.1.\nAt b = 1.04, P = 1, averaging distributions would conceal an extinct low-trust run versus a high-cooperation high-trust run.')
 f,a=setup('A 41.86% mean conceals extinction versus high cooperation','Diagnostic setting: b = 1.04, P = 1 | final strategy grids and observed cooperation trajectories.',1,3,(16,6.8))
 for ax,idx in zip(a[0,:2],ids(1.04,1.)):
  grid=pd.read_csv(SRC/S.loc[idx,'output_dir']/'final_grid.csv',header=None).eq('C').to_numpy();ax.imshow(grid,cmap=ListedColormap(['#e9edf0',TEAL]),vmin=0,vmax=1,interpolation='nearest');ax.set_title(f'Seed {int(S.loc[idx,"seed"])} | final C = {grid.mean()*100:.2f}%');ax.set(xticks=[],yticks=[],xlabel='50 × 50 periodic lattice')
 ax=a[0,2]
 for idx,c in zip(ids(1.04,1.),[TEAL,ORANGE]):ax.plot(T,H['cooperation_rate'][idx]*100,color=c,lw=1.3,label=f'Seed {int(S.loc[idx,"seed"])}')
 ax.axvspan(9001,10000,color='#dfe8ee',zorder=-1);ax.set(xlabel='Round',ylabel='Cooperation (%)',ylim=(-2,103),xlim=(0,10000),title='Two full trajectories');ax.legend(fontsize=9);ax.grid(alpha=.5)
 save(f,'07_seed_dependence','Grid colors: teal = cooperation; light gray = defection. Final-window C: seed 42 = 0.00%; seed 43 = 83.72%.\nTwo seeds cannot estimate the probability of either outcome or establish a bistable phase.')
 f,a=setup('Cooperation gains accompany higher final-window payoffs','Per-person payoff differences relative to the paired Fermi control; actual round-end payoffs, without smoothing.',2,2,(15,8.5))
 for ax,b in zip(a.flat,BS[:4]):
  for group,label,c in [('population','Population',NAVY),('eligible','Eligible positions',TEAL),('ineligible','Ineligible positions',ORANGE)]:
   col=f'delta_{group}_tail_mean_payoff';g=S[S.b.eq(b)].groupby('llm_access_fraction')[col].agg(['mean','min','max']);ax.plot(g.index,g['mean'],color=c,label=label,lw=1.7,marker='o',markersize=3);ax.fill_between(g.index,g['min'],g['max'],color=c,alpha=.10)
  ax.set(title=f'b = {b:g}',xlabel='Advice eligibility P',ylabel='Payoff gain / person / round',xlim=(-.02,1.02),xticks=np.arange(0,1.1,.2));ax.axhline(0,color=GRAY,lw=.6);ax.grid(alpha=.5)
 a[0,0].legend(fontsize=9)
 save(f,'08_payoff','Groups are fixed eligibility positions; empty groups have no mean. Differences include neighborhood spillovers.\nBenefits are not uniform over time: 21/70 active cells have a negative first-200-round mean payoff difference; 10/70 over the full horizon.')
 idx=int(ids(1.01,.7)[0]);start=int(np.flatnonzero(H['cooperation_rate'][idx]==1)[0]);sel=np.arange(start,10001)
 f,a=setup('Constant payoff can still generate positive trust feedback','Diagnostic run: b = 1.01, P = 0.7, seed 42 | feedback uses the sign of payoff minus previous aspiration.',1,3,(16,6.7))
 ax=a[0,0];ax.plot(sel,H['average_payoff'][idx,sel],color=TEAL,label='Population payoff');ax.plot(sel,H['eligible_aspiration_mean'][idx,sel],color=ORANGE,ls='--',label='Eligible aspiration');ax.set(ylabel='Payoff / aspiration',title=f'(a) All C from round {start}',ylim=(3.95,4.01));ax.legend(fontsize=9)
 ax=a[0,1]
 for col,label,c in [('trust_positive_feedback','Positive',TEAL),('trust_negative_feedback','Negative',ORANGE),('trust_zero_feedback','Zero',GRAY)]:ax.plot(sel,H[col][idx,sel]/1750*100,color=c,label=label,lw=1.6)
 ax.set(ylabel='Events / eligible population (%)',title='(b) Feedback stays positive',ylim=(-2,103));ax.legend(fontsize=9)
 ax=a[0,2];ax.plot(sel,H['eligible_trust_mean'][idx,sel],color=TEAL);ax.set(ylabel='Mean eligible trust',title='(c) Trust saturates near one',ylim=(.9,1.003))
 for ax in a[0]:ax.set_xlabel('Round');ax.grid(alpha=.5);ax.set_xlim(start,10000);ax.set_xticks([start,8500,10000],[str(start),'8,500','10,000']);ax.get_xticklabels()[-1].set_ha('right')
 save(f,'09_feedback_caveat','Final aspiration = 3.9999999999999982 versus payoff = 4.0. Last 1,000 rounds: 1,750,000 positive events; 0 negative; 0 zero.\nThe rule rewards exceeding aspiration, not a new payoff increase. Aspiration lag and strict sign comparison motivate a tolerance sensitivity test.')
 frame=S[['b','llm_access_fraction','seed','tail_mean_cooperation_rate','control_tail_mean_cooperation_rate','tail_mean_eligible_trust','tail_mean_eligible_advice_rate','final_eligible_low_trust_fraction','final_eligible_high_trust_fraction']].copy();frame['fixed_05_tail_cooperation']=OLD.tail_mean_cooperation_rate;frame['adaptive_minus_fixed_05']=frame.tail_mean_cooperation_rate-frame.fixed_05_tail_cooperation;frame.to_csv(OUT/'data/figure_values.csv',index=False)
 print('Created 9 figures in PNG, PDF and SVG:',FIG)
if __name__=='__main__':main()
