import pandas as pd, numpy as np, json, sys
sys.path.insert(0,'.')
from common.models import CONSISTENT_MODELS
N={k:v['conversion_factor'] for k,v in json.load(open('resources/nielsen_ome.json')).items()}
base={'oral_morphine':1,'oral_oxycodone':1.5,'oral_oxymorphone':3,'oral_methadone':4.7,'oral_hydrocodone':1,'oral_codeine':0.15,'oral_tapentadol':0.4}
S={
 'CDC2016':{**base,'oral_hydromorphone':4,'oral_tramadol':0.1,'transdermal_fentanyl':2.4},
 'CDC2022':{**base,'oral_hydromorphone':5,'oral_tramadol':0.2,'transdermal_fentanyl':2.4},
 'HEAL':{**base,'oral_hydromorphone':5,'oral_tramadol':0.2,'transdermal_fentanyl':2.4,'oral_tapentadol':0.3,'sublingual_buprenorphine':38.8,'transdermal_buprenorphine':2.2},
 'McPherson':{'oral_morphine':1,'iv_morphine':2.5,'oral_oxycodone':1.25,'oral_hydrocodone':1,'oral_hydromorphone':5,'iv_hydromorphone':12.5,'oral_oxymorphone':2.5,'iv_fentanyl':0.167},
}
S['McPherson_classic']={'oral_morphine':1,'iv_morphine':3,'oral_oxycodone':1.5,'oral_hydrocodone':1,'oral_hydromorphone':4,'iv_hydromorphone':20,'oral_oxymorphone':3,'iv_fentanyl':0.3}
d=pd.read_csv('results/experiment_results.csv')
d=d[(d.include_nielsen==False)&(d.include_examples==False)&d.model.isin(CONSISTENT_MODELS)&(d.ratio_observed_expected>0)]
d['lr']=np.log(d.ratio_observed_expected)
g=d.groupby(['from_opioid','to_opioid','model']).lr.agg(['median','std','count']).reset_index()
tol=np.log(1.10); tight=np.log(1.15)
rows=[]
for (f,t),pg in g.groupby(['from_opioid','to_opioid']):
    rel={s:np.log((v[f]/v[t])/(N[f]/N[t])) for s,v in S.items() if f in v and t in v}
    for _,r in pg.iterrows():
        wrong=abs(r['median'])>tight
        others=pg[(pg.model!=r.model)&(abs(pg['median'])>tight)]['median']
        rows.append(dict(pair=f+'>'+t,model=r.model,wrong=wrong,within_sd=r['std'],
          n_std=len(rel),
          match_std=any(abs(r['median']-x)<tol for x in rel.values()) and wrong,
          match_altstd_nonN=any(abs(r['median']-x)<tol and abs(x)>tol for x in rel.values()) and wrong,
          n_wrong_peers_agree=int((abs(others-r['median'])<tol).sum()) if wrong else np.nan))
R=pd.DataFrame(rows)
print('model-pair cells',len(R),'wrong (outside tight):',R.wrong.sum())
W=R[R.wrong]
print('wrong cells in pairs with >=1 alt standard:',(W.n_std>0).sum())
print('  ...matching some alt standard within 10% (that differs from Nielsen):',W[W.n_std>0].match_altstd_nonN.sum())
print('wrong cells sharing value (+-10%) with >=1 other wrong model:',(W.n_wrong_peers_agree>=1).mean().round(3),' with >=3:',(W.n_wrong_peers_agree>=3).mean().round(3))
by=g.groupby(['from_opioid','to_opioid']).agg(within=('std','median'),between=('median','std'))
print('median within-model SD (log):',round(by.within.median(),3),' median between-model SD of model medians (log):',round(by.between.median(),3))
print('pairs where between > within:',(by.between>by.within).mean().round(3))
print(W[W.match_altstd_nonN].groupby('pair').model.count().sort_values(ascending=False).head(10))
