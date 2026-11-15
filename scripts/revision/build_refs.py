import pandas as pd, numpy as np, sys
REPO, SHEET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sh = pd.read_csv(SHEET)
sh = sh.rename(columns={sh.columns[0]:'opioid_route', sh.columns[1]:'importance_rank', 'CDC RATIO':'sheet_CDC_RATIO_as_entered','NIH HEAL':'NIH_HEAL'})
sh = sh[sh.opioid_route.notna() & sh.opioid_route.str.contains('_', na=False)].drop(columns=[c for c in sh.columns if c.startswith('Unnamed')])
sh = sh.rename(columns={'conversion_factor':'Nielsen_median','LQ':'Nielsen_LQ','UQ':'Nielsen_UQ'})
cdc2016 = {'oral_codeine':0.15,'transdermal_fentanyl':2.4,'oral_hydrocodone':1,'oral_hydromorphone':4,'oral_morphine':1,
           'oral_oxycodone':1.5,'oral_oxymorphone':3,'oral_tapentadol':0.4,'oral_tramadol':0.1}
cdc2022 = {'oral_codeine':0.15,'transdermal_fentanyl':2.4,'oral_hydrocodone':1,'oral_hydromorphone':5,'oral_methadone':4.7,
           'oral_morphine':1,'oral_oxycodone':1.5,'oral_oxymorphone':3,'oral_tapentadol':0.4,'oral_tramadol':0.2}
# McPherson 2018 (ASHP, 2nd ed) equianalgesic doses converted to oral-morphine equivalents; 10 mg IV morphine = 25 mg PO morphine
mcph = {'oral_morphine':1,'iv_morphine':2.5,'oral_oxycodone':1.25,'oral_hydrocodone':1,'oral_hydromorphone':5,
        'iv_hydromorphone':12.5,'oral_oxymorphone':2.5,'parenteral_oxymorphone':25,'iv_fentanyl':25/150}
sh['CDC_2016'] = sh.opioid_route.map(cdc2016)
sh['CDC_2022'] = sh.opioid_route.map(cdc2022)
sh['McPherson_2018'] = sh.opioid_route.map(mcph).round(3)
STD = ['Nielsen_median','CDC_2016','CDC_2022','NIH_HEAL','McPherson_2018']
for c in STD+['Nielsen_LQ','Nielsen_UQ','sheet_CDC_RATIO_as_entered']: sh[c]=pd.to_numeric(sh[c],errors='coerce')
v = sh[STD]
sh['n_standards'] = v.notna().sum(1)
sh['min_factor'] = v.min(1); sh['max_factor'] = v.max(1)
sh['fold_spread_max_over_min'] = (sh.max_factor/sh.min_factor).round(3)
sh.loc[sh.n_standards<2,'fold_spread_max_over_min'] = np.nan
notes = {
 'oral_methadone':'CDC 2016 used dose-tiered factors (4 to 12), not a single value; left blank. Fixed-ratio conversion onto methadone is discouraged.',
 'transdermal_fentanyl':'Factor per mcg/hr. Sheet CDC value 7.2 appears to be CDC prescription-data (NDC) scaling for a 72-hour patch (2.4 x 3), not the guideline factor.',
 'transdermal_buprenorphine':'Factor per mcg/hr. Sheet CDC value 12.6 appears to be NDC-file scaling for a 7-day patch; CDC 2022 guideline lists no buprenorphine factor.',
 'sublingual_buprenorphine':'CDC 2022 guideline lists no buprenorphine factor; sheet CDC value 30 appears to come from the NDC file.',
 'oral_hydromorphone':'Sheet CDC value 4 is the 2016 factor; CDC 2022 changed it to 5.',
 'oral_tramadol':'Sheet CDC value 0.1 is the 2016 factor; CDC 2022 changed it to 0.2.',
 'iv_fentanyl':'Per mcg. McPherson: 0.15 mg IV fentanyl = 10 mg IV morphine = 25 mg PO morphine.',
 'iv_hydromorphone':'McPherson revised IV hydromorphone to 2 mg = 10 mg IV morphine.',
 'oral_pentazocine':'HEAL and sheet CDC value 0.37 (CDC value not in the 2022 guideline table).',
 'sublingual_fentanyl':'Sheet CDC value 0.13 not in the 2022 guideline table.',
}
sh['notes'] = sh.opioid_route.map(notes)
cols = ['opioid_route','importance_rank','administration_method','substance','unit','Nielsen_median','Nielsen_LQ','Nielsen_UQ',
        'CDC_2016','CDC_2022','NIH_HEAL','McPherson_2018','n_standards','min_factor','max_factor','fold_spread_max_over_min',
        'sheet_CDC_RATIO_as_entered','notes']
sh[cols].to_csv(f'{OUT}/reference_standards_by_agent.csv', index=False)
# ---- per-conversion comparison for the study's scored conversions
F = {s: dict(zip(sh.opioid_route, sh[s])) for s in STD+['Nielsen_LQ','Nielsen_UQ']}
def expd(s, froms, doses, to):
    f=F[s]; vals=[f.get(x) for x in froms]+[f.get(to)]
    if any(pd.isna(x) for x in vals): return np.nan
    return sum(d*f[x] for x,d in zip(froms,doses))/f[to]
rows=[]
rd = pd.read_csv(f'{REPO}/results/experiment_results.csv')[['from_opioid','to_opioid']].drop_duplicates()
for _,r in rd.iterrows():
    rows.append(dict(experiment='random_dose', conversion_id=f"{r.from_opioid} -> {r.to_opioid}",
        froms=[r.from_opioid], doses=[1.0], to=r.to_opioid,
        conversion_type='to_OME' if r.to_opioid=='oral_morphine' else 'opioid_to_opioid'))
vg = pd.read_csv(f'{REPO}/results/reworded_vignettes_results.csv').groupby('scenario_key')[['from_opioid','from_dose','to_opioid']].first()
for k,r in vg.iterrows():
    fr=r.from_opioid.split(','); ds=[float(x) for x in str(r.from_dose).split(',')]
    rows.append(dict(experiment='vignette', conversion_id=k, froms=fr, doses=ds if len(fr)>1 else [1.0], to=r.to_opioid,
        conversion_type='to_OME' if r.to_opioid=='oral_morphine' else 'opioid_to_opioid'))
out=[]
for r in rows:
    base=expd('Nielsen_median',r['froms'],r['doses'],r['to'])
    rec=dict(experiment=r['experiment'],conversion=r['conversion_id'],from_opioid=','.join(r['froms']),to_opioid=r['to'],
             conversion_type=r['conversion_type'])
    for s in STD[1:]:
        e=expd(s,r['froms'],r['doses'],r['to']); rec[f'{s}_vs_Nielsen']=round(e/base,3) if pd.notna(e) else np.nan
    # Nielsen IQR envelope on the pair (low source / high target and vice versa)
    lo=[];hi=[]
    for q1,q2 in [('Nielsen_LQ','Nielsen_UQ'),('Nielsen_UQ','Nielsen_LQ')]:
        f1,f2=F[q1],F[q2]
        try:
            num=sum(d*(f1[x] if pd.notna(f1[x]) else F['Nielsen_median'][x]) for x,d in zip(r['froms'],r['doses']))
            den=f2[r['to']] if pd.notna(f2[r['to']]) else F['Nielsen_median'][r['to']]
            (lo if q1=='Nielsen_LQ' else hi).append(num/den/base)
        except KeyError: pass
    rec['Nielsen_IQR_low_vs_median']=round(min(lo+hi),3) if lo+hi else np.nan
    rec['Nielsen_IQR_high_vs_median']=round(max(lo+hi),3) if lo+hi else np.nan
    alts=[rec[f'{s}_vs_Nielsen'] for s in STD[1:] if pd.notna(rec[f'{s}_vs_Nielsen'])]
    rec['n_alternative_standards']=len(alts)
    allv=alts+[1.0]
    rec['envelope_low']=min(allv); rec['envelope_high']=max(allv)
    rec['envelope_fold_width']=round(max(allv)/min(allv),3)
    far=max([max(a,1/a) for a in alts]) if alts else np.nan
    rec['max_fold_difference_from_Nielsen']=round(far,3) if alts else np.nan
    rec['any_standard_outside_tight_band']= (any(a<0.85 or a>1.15 for a in alts)) if alts else np.nan
    rec['any_standard_outside_broad_band']= (any(a<0.6 or a>1.7 for a in alts)) if alts else np.nan
    out.append(rec)
P=pd.DataFrame(out).sort_values(['experiment','conversion_type','conversion'])
P.to_csv(f'{OUT}/reference_standards_by_conversion.csv', index=False)
print(sh[cols[:1]+STD+['fold_spread_max_over_min']].dropna(subset=['fold_spread_max_over_min']).to_string(index=False))
print(P.groupby('experiment').agg(n=('conversion','size'),covered=('n_alternative_standards',lambda s:(s>0).sum()),
      outside_tight=('any_standard_outside_tight_band',lambda s:(s==True).sum()),outside_broad=('any_standard_outside_broad_band',lambda s:(s==True).sum())))
