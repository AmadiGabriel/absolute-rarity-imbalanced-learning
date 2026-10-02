
"""Leak-free repeated nested-CV benchmark on native real-data distributions."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from study_common import METHODS, MODELS, make_pipeline, metric_dict, load_local_dataset, stable_seed, write_manifest

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'; RESULTS=ROOT/'results'; RESULTS.mkdir(exist_ok=True)
DATASETS={k:DATA/f'{k}.csv' for k in ['creditcard','mammography','htru2','aps_failure','secom','oil_spill']}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--profile',choices=['pilot','full'],default='pilot')
    p.add_argument('--datasets',nargs='+',choices=DATASETS.keys(),default=['mammography','htru2','secom','oil_spill'])
    p.add_argument('--models',nargs='+',choices=MODELS,default=['logistic','random_forest'])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS)); return p.parse_args()

def param_grid(model):
    if model=='logistic': return {'clf__C':[0.1,1.0,10.0]}
    if model=='random_forest': return {'clf__max_features':['sqrt',0.5],'clf__min_samples_leaf':[1,3,8]}
    if model=='svm_rbf': return {'clf__C':[0.5,1.0,5.0],'clf__gamma':['scale',0.01]}
    raise ValueError(model)

def main():
    args=parse_args(); cfg_all=json.loads((ROOT/'config.json').read_text()); cfg=cfg_all['real']; base=cfg_all['random_seed']
    k_req=cfg['pilot_outer_splits'] if args.profile=='pilot' else cfg['full_outer_splits']
    r_req=cfg['pilot_outer_repeats'] if args.profile=='pilot' else cfg['full_outer_repeats']
    rows=[]; failures=[]
    for dsi,dname in enumerate(args.datasets):
        path=DATASETS[dname]
        if not path.exists(): print(f'SKIP {dname}: run prepare_data.py first'); continue
        X,y=load_local_dataset(path); nmin=int(y.sum()); k=max(2,min(k_req,nmin))
        outer=RepeatedStratifiedKFold(n_splits=k,n_repeats=r_req,random_state=stable_seed(base,'outer',dname))
        print(f'\n{dname}: n={len(y):,}, p={X.shape[1]}, minority={nmin}, k={k}, repeats={r_req}')
        for outer_index,(tr,te) in enumerate(outer.split(X,y)):
            repeat=outer_index//k; fold=outer_index%k
            Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y[tr],y[te]; mtr=int(ytr.sum()); mte=int(yte.sum())
            k_inner=max(2,min(cfg['inner_splits'],mtr))
            inner=StratifiedKFold(n_splits=k_inner,shuffle=True,random_state=stable_seed(base,'inner',dname,repeat,fold))
            for model in args.models:
                for method in args.methods:
                    seed=stable_seed(base,'realfit',dname,repeat,fold,model,method)
                    try:
                        pipe=make_pipeline(model,method,seed,y_train=ytr)
                        search=GridSearchCV(pipe,param_grid(model),scoring='roc_auc',cv=inner,n_jobs=-1,refit=True,error_score=np.nan)
                        search.fit(Xtr,ytr); met=metric_dict(search.best_estimator_,Xte,yte)
                        rows.append({'dataset':dname,'outer_repeat':repeat,'outer_fold':fold,'outer_index':outer_index,
                                     'outer_k':k,'outer_repeats':r_req,'n':len(y),'p':X.shape[1],
                                     'minority_total':nmin,'prevalence_total':nmin/len(y),
                                     'n_train':len(ytr),'m_train':mtr,'pi_train':mtr/len(ytr),
                                     'n_test':len(yte),'m_test':mte,'pi_test':mte/len(yte),
                                     'model':model,'method':method,'best_inner_auc':search.best_score_,
                                     'best_params':repr(search.best_params_),**met})
                    except Exception as e:
                        failures.append({'dataset':dname,'outer_repeat':repeat,'outer_fold':fold,
                                         'm_train':mtr,'m_test':mte,'model':model,'method':method,'error':repr(e)})
            print(f'  repeat {repeat+1}/{r_req}, fold {fold+1}/{k} complete')
    pd.DataFrame(rows).to_csv(RESULTS/f'real_nested_{args.profile}.csv',index=False)
    pd.DataFrame(failures).to_csv(RESULTS/f'real_nested_failures_{args.profile}.csv',index=False)
    write_manifest(RESULTS/f'real_nested_manifest_{args.profile}.json',cfg_all,{'datasets':args.datasets,'models':args.models,'methods':args.methods})
if __name__=='__main__': main()
