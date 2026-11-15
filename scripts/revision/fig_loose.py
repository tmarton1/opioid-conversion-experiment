import pandas as pd, numpy as np, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
REPO,OUT=sys.argv[1],sys.argv[2]
sys.path.insert(0,REPO); import os; os.chdir(REPO)
from common.models import CONSISTENT_MODELS
P=pd.read_csv(f'{OUT}/reference_standards_by_conversion.csv')
R=pd.read_csv(f'{REPO}/results/experiment_results.csv')
R=R[(R.include_nielsen==False)&(R.include_examples==False)&R.model.isin(CONSISTENT_MODELS)&(R.ratio_observed_expected>0)]
INK='#2b2b2a'; MUTED='#6b6a65'; GRID='#e4e3df'; BLUE='#2a78d6'; ORANGE='#eb6834'
ALT=[('CDC_2022_vs_Nielsen','CDC 2022'),('NIH_HEAL_vs_Nielsen','HEAL'),('McPherson_2018_vs_Nielsen','McPherson')]
EX=[('oral_hydrocodone','oral_morphine','Oral hydrocodone → oral morphine','Nielsen is the outlier: every other standard gives 0.83×'),
    ('iv_hydromorphone','oral_morphine','IV hydromorphone → oral morphine','One alternative (McPherson), 29% below Nielsen'),
    ('oral_oxycodone','oral_morphine','Oral oxycodone → oral morphine','Nielsen sits inside the other standards (0.83–1×)'),
    ('oral_codeine','oral_morphine','Oral codeine → oral morphine','Nielsen is the outlier: CDC and HEAL give 1.5×')]
fold=lambda r,s: np.min(np.abs(np.log(np.asarray(r)[:,None]/np.asarray(s)[None,:])),axis=1)
x=np.exp(np.linspace(np.log(0.25),np.log(4),900))
fig=plt.figure(figsize=(11,11),facecolor='white')
outer=fig.add_gridspec(2,2,hspace=0.42,wspace=0.16,left=0.075,right=0.985,top=0.80,bottom=0.085)
summ=[]
for k,(f,t,title,sub) in enumerate(EX):
    row=(k//2)*2; col=k%2
    inner=outer[k//2,col].subgridspec(2,1,height_ratios=[3,1],hspace=0.06)
    ax=fig.add_subplot(inner[0]); hx=fig.add_subplot(inner[1],sharex=ax)
    p=P[(P.experiment=='random_dose')&(P.conversion==f'{f} -> {t}')].iloc[0]
    alts=sorted({p[c] for c,_ in ALT if pd.notna(p[c])})
    curves=[('Nielsen only',[1.0],INK,'-'),('All standards',sorted(set(alts+[1.0])),BLUE,'--'),('All except Nielsen',alts,ORANGE,':')]
    for lab,S,c,ls in curves:
        ax.plot(x,np.exp(fold(x,S)),color=c,ls=ls,lw=2.2 if ls!=':' else 2.6,label=lab,zorder=3)
    for yv,l in [(1.15,'tight'),(1.7,'broad')]:
        ax.axhline(yv,color=MUTED,lw=0.8,ls=(0,(4,3)),zorder=1); ax.text(3.9,yv*1.025,l,ha='right',va='bottom',fontsize=7.5,color=MUTED)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.25,4); ax.set_ylim(1,4.2)
    ax.set_yticks([1,1.15,1.7,2,3,4]); ax.set_yticklabels(['1×','1.15×','1.7×','2×','3×','4×'],fontsize=8,color=MUTED); ax.minorticks_off()
    plt.setp(ax.get_xticklabels(),visible=False)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(GRID)
    ax.grid(axis='y',color=GRID,lw=0.6,zorder=0)
    # standards as labeled ticks
    ax.axvline(1,color=INK,lw=0.8,alpha=.25,zorder=0)
    names={}
    for c_,n in ALT:
        if pd.notna(p[c_]): names.setdefault(round(p[c_],3),[]).append(n)
    names.setdefault(1.0,[]).insert(0,'Nielsen')
    for v,ns in names.items():
        ax.plot([v],[1.0],marker='v',ms=8,color=INK if 'Nielsen' in ns and len(ns)==1 else ORANGE,mec=INK,mew=.6,clip_on=False,zorder=5)
        ax.annotate(' / '.join(ns),(v,1.0),xytext=(0,-2),textcoords='offset points',ha='center',va='top',fontsize=7.2,color=INK,annotation_clip=False,zorder=6) if False else None
    items=sorted(names.items())
    for j,(v,ns) in enumerate(items):
        left = (j==0 and len(items)>1)
        ax.text(v*(0.96 if left else (1.04 if len(items)>1 else 1)),1.3,' / '.join(ns),ha='right' if left else ('left' if len(items)>1 else 'center'),
                fontsize=7.6,color=INK,bbox=dict(boxstyle='round,pad=0.2',fc='white',ec=GRID,lw=0.6),zorder=6)
    ax.set_title(title,loc='left',fontsize=10.5,fontweight='bold',color=INK,pad=16)
    ax.text(0,1.015,sub,transform=ax.transAxes,fontsize=8.3,color=MUTED,va='bottom')
    # real naive responses
    r=R[(R.from_opioid==f)&(R.to_opioid==t)].ratio_observed_expected.clip(0.26,3.9).values
    bins=np.exp(np.linspace(np.log(0.25),np.log(4),49))
    hx.hist(r,bins=bins,color='#b9b8b2',edgecolor='white',lw=0.5)
    hx.set_xscale('log'); hx.set_xlim(0.25,4); hx.set_yticks([])
    hx.set_xticks([0.25,0.5,0.85,1,1.15,2,4]); hx.set_xticklabels(['¼×','½×','0.85×','1×','1.15×','2×','4×'],fontsize=8,color=MUTED); hx.minorticks_off()
    for s in ['top','right','left']: hx.spines[s].set_visible(False)
    hx.spines['bottom'].set_color(GRID)
    tn=np.mean(fold(r,[1.0])<=np.log(1.15)); ta=np.mean(fold(r,sorted(set(alts+[1.0])))<=np.log(1.15)); tx=np.mean(fold(r,alts)<=np.log(1.15))
    hx.text(0.0,-0.38,f'Naive responses (n = {len(r)}) · within tight:  Nielsen {tn:.0%}   All {ta:.0%}   All except Nielsen {tx:.0%}',
            transform=hx.transAxes,fontsize=8,color=INK,va='top')
    if col==0: ax.set_ylabel('Fold error assigned',fontsize=9,color=INK)
fig.suptitle('How a response is scored under the three reference definitions',x=0.01,ha='left',y=0.985,fontsize=13.5,fontweight='bold',color=INK)
fig.text(0.01,0.947,'A response that matches any standard in the set scores 1× (no error); otherwise its error is the fold distance to the nearest standard in the set.\n'
         'Standards: Nielsen, CDC 2022, NIH HEAL, McPherson 2018 (CDC 2016 excluded as superseded). Curves show the error each score assigns to every possible answer;\n'
         'grey histograms show the actual naive random-dose responses from the 13 models for that conversion.',fontsize=8.6,color=MUTED,va='top')
h=[Line2D([0],[0],color=INK,lw=2.2,label='Nielsen only'),Line2D([0],[0],color=BLUE,lw=2.2,ls='--',label='All standards'),
   Line2D([0],[0],color=ORANGE,lw=2.6,ls=':',label='All except Nielsen'),
   Line2D([0],[0],ls='none',marker='v',ms=8,color=ORANGE,mec=INK,mew=.6,label='Standard (black: Nielsen alone)')]
fig.text(0.53,0.008,'Model response relative to the Nielsen expected dose (log scale)',ha='center',fontsize=9.5,color=INK)
fig.legend(handles=h,loc='upper left',bbox_to_anchor=(0.01,0.875),ncol=4,frameon=False,fontsize=9)
fig.savefig(f'{OUT}/fig_loose_reference_scoring.png',dpi=220,facecolor='white')
