import pandas as pd, numpy as np, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
OUT=sys.argv[1]
A=pd.read_csv(f'{OUT}/reference_standards_by_agent.csv'); P=pd.read_csv(f'{OUT}/reference_standards_by_conversion.csv')
INK='#2b2b2a'; MUTED='#6b6a65'; GRID='#e4e3df'; SURF='#ffffff'
TIGHT='#dcebdc'; BROAD='#f1f6f1'
STD=[('CDC_2016','CDC 2016 (superseded)',dict(marker='o',mfc='white',mec='#8a8984',ms=7.5,mew=1.6)),
     ('CDC_2022','CDC 2022',dict(marker='s',mfc='#2a78d6',mec=INK,ms=7.5,mew=0.6)),
     ('NIH_HEAL','NIH HEAL',dict(marker='^',mfc='#eb6834',mec=INK,ms=8.5,mew=0.6)),
     ('McPherson_2018','McPherson 2018',dict(marker='D',mfc='#1baf7a',mec=INK,ms=7,mew=0.6))]
OFF=np.linspace(-0.27,0.27,4)
XT=[0.5,0.6,0.85,1,1.15,1.7,2]; XL=['0.5×','0.6×','0.85×','1×','1.15×','1.7×','2×']
def nice(k):
    rt,dr=k.split('_',1)
    rt={'oral':'Oral','iv':'IV','sublingual':'SL','transdermal':'Patch','sc':'SC','im':'IM','parenteral':'Parenteral'}[rt]
    return f'{dr.capitalize()} ({rt})'
def base_axes(ax,n):
    ax.set_xscale('log'); ax.set_xlim(0.45,2.25)
    ax.axvspan(0.6,1.7,color=BROAD,zorder=0,lw=0); ax.axvspan(0.85,1.15,color=TIGHT,zorder=0,lw=0)
    ax.axvline(1,color=INK,lw=1.4,zorder=1)
    ax.set_xticks(XT); ax.set_xticklabels(XL,fontsize=9,color=MUTED); ax.minorticks_off()
    ax.set_ylim(n-0.5,-0.5); ax.tick_params(axis='y',length=0)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    for y in range(n): ax.axhline(y+0.5,color=GRID,lw=0.6,zorder=0)
def legend(fig,y):
    h=[Line2D([0],[0],color=INK,lw=1.4,label='Nielsen median (reference)'),
       Line2D([0],[0],color='#9a9993',lw=5,alpha=.7,label='Nielsen interquartile range')]
    h+= [Line2D([0],[0],ls='none',label=l,**kw) for _,l,kw in STD]
    h+= [Patch(color=TIGHT,label='Tight band (0.85–1.15×)'),Patch(color=BROAD,label='Broad band (0.6–1.7×)')]
    fig.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,y),ncol=4,frameon=False,fontsize=8.5,handlelength=1.6,columnspacing=1.2)
# ---------- Figure A: per agent
study=set(P.from_opioid.str.split(',').explode())|set(P.to_opioid)
alt=[s for s,_,_ in STD]
a=A[A.opioid_route.isin(study)&(A.opioid_route!='oral_morphine')].copy()
rel=a[alt].div(a.Nielsen_median,axis=0)
a['has_alt']=rel.notna().any(axis=1)
a['maxdev']=np.exp(np.log(rel).abs().max(axis=1))
noalt=a[~a.has_alt]; a=a[a.has_alt].sort_values('maxdev',ascending=False).reset_index(drop=True)
rel=a[alt].div(a.Nielsen_median,axis=0)
n=len(a); n_out=int(((rel<0.85)|(rel>1.15)).any(axis=1).sum())
rc=rel[['CDC_2022','NIH_HEAL','McPherson_2018']]; n_out_cur=int(((rc<0.85)|(rc>1.15)).any(axis=1).sum())
fig,ax=plt.subplots(figsize=(8.2,0.48*n+2.6),facecolor=SURF)
base_axes(ax,n)
for i,r in a.iterrows():
    if pd.notna(r.Nielsen_LQ) and pd.notna(r.Nielsen_UQ) and r.Nielsen_UQ>r.Nielsen_LQ:
        ax.plot([r.Nielsen_LQ/r.Nielsen_median,r.Nielsen_UQ/r.Nielsen_median],[i,i],color='#9a9993',lw=5,alpha=.7,solid_capstyle='butt',zorder=2)
    for (s,_,kw),o in zip(STD,OFF):
        v=rel.loc[i,s]
        if pd.notna(v): ax.plot(v,i+o,ls='none',zorder=3,**kw)
ax.set_yticks(range(n)); ax.set_yticklabels([f"{nice(k)}  ·  Nielsen {v:g}" for k,v in zip(a.opioid_route,a.Nielsen_median)],fontsize=9,color=INK)
ax.set_xlabel('Conversion factor relative to the Nielsen median (log scale)',fontsize=9.5,color=INK)
fig.suptitle(f'Published standards differ from Nielsen by more than ±15% for {n_out} of {n} agents',x=0.02,ha='left',fontsize=12.5,fontweight='bold',color=INK,y=0.995)
fig.text(0.02,0.962,f'Oral morphine equivalent factor for each agent in the study, under each standard ({n_out_cur} of {n} excluding the superseded CDC 2016 factors).\nAgents are sorted by the largest difference from Nielsen.',va='top',fontsize=8.8,color=MUTED)
legend(fig,0.915)
fig.text(0.02,0.012,'No alternative standard available: '+', '.join(nice(k) for k in noalt.opioid_route)+'. McPherson values are from reprinted tables and should be checked against the 2018 edition.',fontsize=7.6,color=MUTED,wrap=True)
fig.tight_layout(rect=(0,0.04,1,0.875))
fig.savefig(f'{OUT}/fig_reference_agreement_by_agent.png',dpi=220,facecolor=SURF)
# ---------- Figure B: per scored conversion
Q=P[P.n_alternative_standards>0].copy()
relc=Q[[f'{s}_vs_Nielsen' for s in alt]]
Q['maxdev']=np.exp(np.log(relc).abs().max(axis=1))
lab={'random_dose':'Random-dose conversions','vignette':'Clinical vignette scenarios'}
order=[]
for e in ['random_dose','vignette']:
    g=Q[Q.experiment==e].sort_values('maxdev',ascending=False); order.append((e,g))
nrows=sum(len(g) for _,g in order)
fig,axs=plt.subplots(2,1,figsize=(8.6,0.27*nrows+3.4),facecolor=SURF,gridspec_kw=dict(height_ratios=[len(g) for _,g in order],hspace=0.12))
def short(c):
    return c.replace('oral_','PO ').replace('iv_','IV ').replace('transdermal_','patch ').replace('sublingual_','SL ').replace(' -> ',' → ')
tot=len(Q); out_all=int(Q.any_standard_outside_tight_band.sum())
cur=Q[['CDC_2022_vs_Nielsen','NIH_HEAL_vs_Nielsen','McPherson_2018_vs_Nielsen']]
out_cur=int(((cur<0.85)|(cur>1.15)).any(axis=1).sum()); broad=int(Q.any_standard_outside_broad_band.sum())
for ax,(e,g) in zip(axs,order):
    g=g.reset_index(drop=True); m=len(g); base_axes(ax,m)
    for i,r in g.iterrows():
        lo,hi=r.Nielsen_IQR_low_vs_median,r.Nielsen_IQR_high_vs_median
        if hi>lo: ax.plot([lo,hi],[i,i],color='#9a9993',lw=4,alpha=.7,solid_capstyle='butt',zorder=2)
        for (s,_,kw),o in zip(STD,OFF):
            v=r[f'{s}_vs_Nielsen']
            if pd.notna(v): ax.plot(v,i+o*0.8,ls='none',zorder=3,**{**kw,'ms':kw['ms']*0.8})
    ax.set_yticks(range(m)); ax.set_yticklabels([short(c) if e=='random_dose' else c.replace('-',' ').capitalize() for c in g.conversion],fontsize=7.6,color=INK)
    ax.set_title(lab[e],loc='left',fontsize=10,fontweight='bold',color=INK,pad=4)
axs[-1].set_xlabel('Expected dose under each standard relative to the Nielsen expected dose (log scale)',fontsize=9.5,color=INK)
fig.suptitle(f'{out_all} of {tot} scored conversions fall outside the tight band under at least one standard',x=0.02,ha='left',fontsize=12.5,fontweight='bold',color=INK,y=0.997)
fig.text(0.02,0.974,f'{out_cur} of {tot} excluding the superseded CDC 2016 factors. Only tramadol conversions under CDC 2016 exceed the broad band ({broad} conversions).\nOmitted for lack of any alternative standard: 9 random-dose pairs (IV opioids with tramadol, IV/IM buprenorphine, IV methadone,\nnalbuphine, SC fentanyl) and the oral-to-IV methadone vignette.',fontsize=8.4,color=MUTED,va='top')
legend(fig,0.945)
fig.subplots_adjust(left=0.36,right=0.97,top=0.9,bottom=0.045)
fig.savefig(f'{OUT}/fig_reference_agreement_by_conversion.png',dpi=220,facecolor=SURF)
print(n_out,n,out_all,out_cur,tot,broad)
