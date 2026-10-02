
"""
Optional real-data validation-rarity robustness experiment.
Recommended for the larger-minority datasets (creditcard, HTRU2, APS).
A reference holdout pool is kept untouched. Models are trained once per
repetition, then validation subsets of controlled absolute/relative rarity are
drawn from a separate validation pool. The reference holdout provides the
comparison target for the sign/direction of augmentation effects.
"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from study_common import METHODS,MODELS,make_pipeline,metric_dict,load_local_dataset,exact_sample_indices,take_rows,stable_seed,write_manifest
ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
DATASETS={k:DATA/f'{k}.csv' for k in ['creditcard','mammography','htru2','aps_failure','secom','oil_spill']}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--profile',choices=['pilot','full'],default='pilot')
    p.add_argument('--datasets',nargs='+',choices=DATASETS.keys(),default=['creditcard','htru2','aps_failure'])
    p.add_argument('--models',nargs='+',choices=MODELS,default=['logistic','random_forest'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS)); return p.parse_args()

def validation_conditions(cfg):
    N=cfg['validation_fixed_total_n']
    for m in cfg['validation_minority_counts']:
        if m<N: yield 'V1_fixed_total',m,N-m,m/N
    pi=cfg['validation_ratio']
    for m in cfg['validation_minority_counts']:
        maj=int(round(m*(1-pi)/pi)); yield 'V2_fixed_ratio',m,maj,m/(m+maj)
    m=cfg['validation_fixed_minority']
    for pi in cfg['validation_ratios']:
        maj=int(round(m*(1-pi)/pi)); yield 'V3_fixed_minority',m,maj,m/(m+maj)

def main():
    args=parse_args(); cfgall=json.loads((ROOT/'config.json').read_text()); scfg=cfgall['synthetic']; rcfg=cfgall['real']; base=cfgall['random_seed']
    reps=rcfg['validation_rarity_repeats_pilot'] if args.profile=='pilot' else rcfg['validation_rarity_repeats_full']
    rows=[]; refrows=[]; failures=[]
    for dname in args.datasets:
        path=DATASETS[dname]
        if not path.exists(): print('SKIP',dname); continue
        X,y=load_local_dataset(path)
        if int(y.sum())<120:
            print(f'SKIP {dname}: too few minority cases for this optional robustness module')
            continue
        for rep in range(reps):
            # 60% train / 20% validation pool / 20% reference pool, stratified.
            Xtrain,Xhold,ytrain,yhold=train_test_split(X,y,test_size=0.40,stratify=y,random_state=stable_seed(base,'rv1',dname,rep))
            Xvalpool,Xref,yvalpool,yref=train_test_split(Xhold,yhold,test_size=0.50,stratify=yhold,random_state=stable_seed(base,'rv2',dname,rep))
            fitted={}
            for model in args.models:
                for method in args.methods:
                    seed=stable_seed(base,'rvfit',dname,rep,model,method)
                    try:
                        est=make_pipeline(model,method,seed,y_train=ytrain); est.fit(Xtrain,ytrain); fitted[(model,method)]=est
                        met=metric_dict(est,Xref,yref); refrows.append({'dataset':dname,'rep':rep,'model':model,'method':method,
                            'n_train':len(ytrain),'m_train':int(ytrain.sum()),'n_reference':len(yref),'m_reference':int(yref.sum()),**met})
                    except Exception as e: failures.append({'stage':'fit_reference','dataset':dname,'rep':rep,'model':model,'method':method,'error':repr(e)})
            # controlled validation draws from the separate validation pool
            yvp=np.asarray(yvalpool); allidx=np.arange(len(yvp))
            for vs,mv,nmaj,piv in validation_conditions(scfg):
                if mv>int(yvp.sum()) or nmaj>int((1-yvp).sum()): continue
                rng=np.random.default_rng(stable_seed(base,'rvdraw',dname,rep,vs,mv,nmaj))
                idx=exact_sample_indices(yvp,mv,nmaj,rng,allowed_idx=allidx); Xv,yv=take_rows(Xvalpool,yvp,idx)
                for model in args.models:
                    for method in args.methods:
                        est=fitted.get((model,method));
                        if est is None: continue
                        met=metric_dict(est,Xv,yv); rows.append({'dataset':dname,'rep':rep,'validation_scenario':vs,
                            'm_val':mv,'nmaj_val':nmaj,'n_val':mv+nmaj,'pi_val':piv,'model':model,'method':method,**met})
    pd.DataFrame(rows).to_csv(RESULTS/f'real_validation_rarity_{args.profile}.csv',index=False)
    pd.DataFrame(refrows).to_csv(RESULTS/f'real_validation_reference_{args.profile}.csv',index=False)
    pd.DataFrame(failures).to_csv(RESULTS/f'real_validation_failures_{args.profile}.csv',index=False)
    write_manifest(RESULTS/f'real_validation_manifest_{args.profile}.json',cfgall,{'datasets':args.datasets,'models':args.models,'methods':args.methods})
if __name__=='__main__': main()
