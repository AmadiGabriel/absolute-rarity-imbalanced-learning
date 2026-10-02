
"""
Controlled real-data training-rarity experiment.
Within each repetition a fresh stratified test split is held out. From the
remaining development data, training prevalence is held fixed while the number
of genuine minority training observations is varied.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from study_common import METHODS,MODELS,make_pipeline,metric_dict,load_local_dataset,exact_sample,stable_seed,write_manifest
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
DATASETS={k:DATA/f'{k}.csv' for k in ['creditcard','mammography','htru2','aps_failure','secom','oil_spill']}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--profile',choices=['pilot','full'],default='pilot')
    p.add_argument('--datasets',nargs='+',choices=DATASETS.keys(),default=list(DATASETS.keys()))
    p.add_argument('--models',nargs='+',choices=MODELS,default=['logistic','random_forest'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS)); return p.parse_args()

def main():
    args=parse_args(); cfgall=json.loads((ROOT/'config.json').read_text()); cfg=cfgall['real']; base=cfgall['random_seed']
    reps=cfg['subsample_pilot_repeats'] if args.profile=='pilot' else cfg['subsample_full_repeats']
    pi=cfg['controlled_training_prevalence']; test_fraction=cfg['subsample_test_fraction']; mvals=cfgall['synthetic']['minority_counts']
    rows=[]; failures=[]
    for dname in args.datasets:
        path=DATASETS[dname]
        if not path.exists(): print('SKIP',dname); continue
        X,y=load_local_dataset(path)
        for rep in range(reps):
            splitseed=stable_seed(base,'subsplit',dname,rep)
            Xdev,Xtest,ydev,ytest=train_test_split(X,y,test_size=test_fraction,stratify=y,random_state=splitseed)
            rng=np.random.default_rng(stable_seed(base,'subsample',dname,rep))
            for m in mvals:
                nmaj=int(round(m*(1-pi)/pi))
                if m>int(ydev.sum()) or nmaj>int((1-ydev).sum()): continue
                Xtr,ytr=exact_sample(Xdev,ydev,m,nmaj,rng)
                for model in args.models:
                    for method in args.methods:
                        seed=stable_seed(base,'subfit',dname,rep,m,model,method)
                        try:
                            est=make_pipeline(model,method,seed,y_train=ytr); est.fit(Xtr,ytr); met=metric_dict(est,Xtest,ytest)
                            rows.append({'dataset':dname,'rep':rep,'n_train':len(ytr),'m_train':m,'nmaj_train':nmaj,
                                         'pi_train':m/len(ytr),'n_test':len(ytest),'m_test':int(ytest.sum()),
                                         'pi_test':float(np.mean(ytest)),'model':model,'method':method,**met})
                        except Exception as e:
                            failures.append({'dataset':dname,'rep':rep,'m_train':m,'model':model,'method':method,'error':repr(e)})
    pd.DataFrame(rows).to_csv(RESULTS/f'real_subsample_{args.profile}.csv',index=False)
    pd.DataFrame(failures).to_csv(RESULTS/f'real_subsample_failures_{args.profile}.csv',index=False)
    write_manifest(RESULTS/f'real_subsample_manifest_{args.profile}.json',cfgall,{'datasets':args.datasets,'models':args.models,'methods':args.methods})
if __name__=='__main__': main()
