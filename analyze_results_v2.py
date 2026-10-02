
"""
Statistical analysis and plot-data export.

Important: this script works ONLY from saved CSV results. You can change alpha,
equivalence margins, bootstrap repetitions, tables, or plots later without
rerunning any machine-learning models.
"""
from pathlib import Path
import argparse,json,warnings
import numpy as np,pandas as pd
from scipy import stats
from study_common import METRICS, stable_seed
ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results'; OUT=ROOT/'analysis'; PLOT=ROOT/'plot_data'; OUT.mkdir(exist_ok=True); PLOT.mkdir(exist_ok=True)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--profile',choices=['pilot','full'],default='pilot'); return p.parse_args()

def holm(pvals):
    p=np.asarray(pvals,float); out=np.full(len(p),np.nan); ok=np.isfinite(p); vals=p[ok]; idx=np.where(ok)[0]
    if not len(vals): return out
    order=np.argsort(vals); adj=np.empty(len(vals)); running=0
    for rank,j in enumerate(order):
        v=(len(vals)-rank)*vals[j]; running=max(running,v); adj[j]=min(1.0,running)
    out[idx]=adj; return out

def bootstrap_ci(x, reps, alpha, seed):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2: return (np.nan,np.nan)
    rng=np.random.default_rng(seed); means=np.empty(reps)
    for i in range(reps): means[i]=rng.choice(x,size=len(x),replace=True).mean()
    return tuple(np.quantile(means,[alpha/2,1-alpha/2]))

def tost_p(x, margin):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2 or np.std(x,ddof=1)==0: return np.nan
    mean=x.mean(); se=stats.sem(x); df=len(x)-1
    t_low=(mean+margin)/se; p_low=1-stats.t.cdf(t_low,df)
    t_high=(mean-margin)/se; p_high=stats.t.cdf(t_high,df)
    return max(p_low,p_high)

def summarize_delta(df, pair_keys, group_keys, metric, cfg, prefix=''):
    base=df[df.method=='none'][pair_keys+[metric]].rename(columns={metric:'baseline'})
    d=df[df.method!='none'].merge(base,on=pair_keys,how='inner'); d['delta']=d[metric]-d['baseline']
    rows=[]; B=cfg['bootstrap_replicates']; alpha=cfg['alpha']; margin=cfg['equivalence_margins'].get(metric,np.nan)
    for gvals,g in d.groupby(group_keys,dropna=False):
        if not isinstance(gvals,tuple): gvals=(gvals,)
        x=g['delta'].to_numpy(float); n=np.isfinite(x).sum(); mean=np.nanmean(x); sd=np.nanstd(x,ddof=1) if n>1 else np.nan
        lo,hi=bootstrap_ci(x,B,alpha,stable_seed(9137,prefix,metric,*gvals))
        try: wp=stats.wilcoxon(x[np.isfinite(x)],alternative='two-sided').pvalue if np.any(np.abs(x[np.isfinite(x)])>0) else 1.0
        except Exception: wp=np.nan
        row=dict(zip(group_keys,gvals)); row.update({'metric':metric,'n_pairs':n,'mean_delta':mean,'sd_delta':sd,
             'median_delta':np.nanmedian(x),'ci_low':lo,'ci_high':hi,'wilcoxon_p':wp,'equivalence_margin':margin,
             'tost_p':tost_p(x,margin) if np.isfinite(margin) else np.nan,
             'equivalent_by_ci': bool(np.isfinite(margin) and lo>=-margin and hi<=margin) if np.isfinite(lo) else False,
             'cohen_dz': mean/sd if np.isfinite(sd) and sd>0 else np.nan})
        rows.append(row)
    out=pd.DataFrame(rows)
    if len(out):
        out['wilcoxon_p_holm_global']=holm(out['wilcoxon_p'].values)
        fam=[c for c in group_keys if c not in ('method','m_train','m_val','pi_train','pi_val','n_train','n_val')]
        if fam:
            out['wilcoxon_p_holm_family']=np.nan
            for _,idx in out.groupby(fam,dropna=False).groups.items():
                out.loc[idx,'wilcoxon_p_holm_family']=holm(out.loc[idx,'wilcoxon_p'].values)
    return out,d

def corrected_cv_summary(df, metric, cfg):
    pair=['dataset','outer_repeat','outer_fold','model']; base=df[df.method=='none'][pair+[metric]].rename(columns={metric:'baseline'})
    d=df[df.method!='none'].merge(base,on=pair,how='inner'); d['delta']=d[metric]-d['baseline']; rows=[]
    for vals,g in d.groupby(['dataset','model','method'],dropna=False):
        x=g.delta.to_numpy(float); x=x[np.isfinite(x)]; n=len(x); k=int(g.outer_k.iloc[0]); mean=x.mean() if n else np.nan; var=np.var(x,ddof=1) if n>1 else np.nan
        corr=(1/n + 1/(k-1)) if n>1 and k>1 else np.nan; se=np.sqrt(corr*var) if np.isfinite(var) else np.nan
        t=mean/se if np.isfinite(se) and se>0 else np.nan; p=2*stats.t.sf(abs(t),df=n-1) if np.isfinite(t) else np.nan
        crit=stats.t.ppf(1-cfg['alpha']/2,n-1) if n>1 else np.nan
        lo=mean-crit*se if np.isfinite(se) else np.nan; hi=mean+crit*se if np.isfinite(se) else np.nan
        rows.append({'dataset':vals[0],'model':vals[1],'method':vals[2],'metric':metric,'n_outer_folds':n,'outer_k':k,
                     'mean_delta':mean,'corrected_se':se,'corrected_ci_low':lo,'corrected_ci_high':hi,'corrected_t_p':p})
    out=pd.DataFrame(rows)
    if len(out): out['corrected_t_p_holm']=holm(out.corrected_t_p.values)
    return out,d

def validation_reliability(val, oracle, metric):
    """
    Compare validation method-vs-none deltas with the oracle delta.

    AUC: uses the ordinary large-oracle AUC (prevalence invariant).
    AP: if run_synthetic_v2.py supplied oracle_ap_matched, use the oracle AP
        standardized to the SAME validation prevalence. This prevents AP
        reliability from being confounded by a 10% oracle prevalence versus
        a 0.5%-10% validation prevalence.
    """
    val = val.copy()

    if metric == "auc" and "oracle_auc" in val.columns:
        val["oracle_metric"] = val["oracle_auc"]
        v = val
    elif metric == "ap" and "oracle_ap_matched" in val.columns:
        val["oracle_metric"] = val["oracle_ap_matched"]
        v = val
    else:
        # Backward-compatible path for older pilot files.
        o = oracle[oracle.scenario == "A2_fixed_ratio"].copy()
        okeys = ["rep","p","class_sep","flip_y","m_train","pi_train","model","method"]
        oval = o[okeys + [metric]].rename(columns={metric:"oracle_metric"})
        v = val.merge(oval, on=okeys, how="left")

    pair = ["rep","validation_scenario","p","class_sep","flip_y",
            "m_train","pi_train","m_val","pi_val","model"]
    b = v[v.method == "none"][pair + [metric,"oracle_metric"]].rename(
        columns={metric:"v_base","oracle_metric":"o_base"}
    )
    d = v[v.method != "none"].merge(b, on=pair, how="inner")
    d["delta_validation"] = d[metric] - d["v_base"]
    d["delta_oracle"] = d["oracle_metric"] - d["o_base"]
    d["estimation_error"] = d["delta_validation"] - d["delta_oracle"]
    d["absolute_error"] = d["estimation_error"].abs()
    d["sign_error"] = (
        np.sign(d.delta_validation) != np.sign(d.delta_oracle)
    ).astype(int)

    s = (d.groupby(["validation_scenario","model","method","m_val","pi_val"])
           .agg(
               n=("sign_error","size"),
               sign_error_rate=("sign_error","mean"),
               mean_abs_error=("absolute_error","mean"),
               median_abs_error=("absolute_error","median"),
               mean_validation_delta=("delta_validation","mean"),
               mean_oracle_delta=("delta_oracle","mean")
           )
           .reset_index())

    gkeys = ["rep","validation_scenario","p","class_sep","flip_y",
             "m_train","pi_train","m_val","pi_val","model"]
    rows = []
    for key, g in v.groupby(gkeys, dropna=False):
        if g[metric].isna().all() or g.oracle_metric.isna().all():
            continue
        vbest = g.loc[g[metric].idxmax(), "method"]
        obest = g.loc[g.oracle_metric.idxmax(), "method"]
        row = dict(zip(gkeys, key))
        row["validation_best"] = vbest
        row["oracle_best"] = obest
        row["correct_selection"] = int(vbest == obest)
        rows.append(row)

    sel = pd.DataFrame(rows)
    if len(sel):
        sel = (sel.groupby(["validation_scenario","model","m_val","pi_val"])
                  .agg(n=("correct_selection","size"),
                       selection_accuracy=("correct_selection","mean"))
                  .reset_index())
    return s, sel, d


def trend_slopes(syn, metric):
    """
    Primary trend-oriented summaries that respect the repeated-measures design.

    Within each independently generated population cell, fit:
      A2: delta ~ log10(m_train)
      A3: delta ~ log10(pi_train)

    The resulting per-population slopes can later be tested/summarized without
    pretending the five m/ratio cells are independent observations.
    """
    pair = ["rep","scenario","p","class_sep","flip_y",
            "m_train","nmaj_train","pi_train","model"]
    base = syn[syn.method == "none"][pair + [metric]].rename(
        columns={metric:"baseline"}
    )
    d = syn[syn.method != "none"].merge(base, on=pair, how="inner")
    d["delta"] = d[metric] - d["baseline"]

    rows = []
    cluster = ["rep","p","class_sep","flip_y","model","method"]

    for scenario, xcol, transform in [
        ("A2_fixed_ratio", "m_train", lambda x: np.log10(x)),
        ("A3_fixed_minority", "pi_train", lambda x: np.log10(x)),
    ]:
        ds = d[d.scenario == scenario].copy()
        for key, g in ds.groupby(cluster, dropna=False):
            x = transform(g[xcol].to_numpy(float))
            y = g["delta"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.unique(x[ok]).size < 2:
                continue
            slope, intercept = np.polyfit(x[ok], y[ok], 1)
            row = dict(zip(cluster, key))
            row.update({
                "scenario": scenario,
                "metric": metric,
                "x_variable": xcol,
                "slope": float(slope),
                "intercept": float(intercept),
                "n_points": int(ok.sum()),
            })
            rows.append(row)

    return pd.DataFrame(rows)

def main():
    args=parse_args(); cfgall=json.loads((ROOT/'config.json').read_text()); cfg=cfgall['statistics']; profile=args.profile
    synp=RESULTS/f'synthetic_oracle_{profile}.csv'; valp=RESULTS/f'synthetic_validation_{profile}.csv'
    if synp.exists() and synp.stat().st_size>0:
        syn=pd.read_csv(synp); allsum=[]
        pair=['rep','scenario','p','class_sep','flip_y','m_train','nmaj_train','pi_train','model']
        groups=['scenario','p','class_sep','flip_y','m_train','pi_train','model','method']
        for metric in METRICS:
            s,d=summarize_delta(syn,pair,groups,metric,cfg,'synthetic'); allsum.append(s)
            d.to_csv(OUT/f'synthetic_paired_delta_{metric}_{profile}.csv',index=False)
        statsout=pd.concat(allsum,ignore_index=True); statsout.to_csv(OUT/f'synthetic_statistics_{profile}.csv',index=False)

        # Trend-oriented files for primary inferential analysis. These use one
        # slope per independently generated population cell and avoid treating
        # repeated m_train/pi_train conditions as independent.
        trend_frames=[]
        for metric in ("auc","ap"):
            tr=trend_slopes(syn,metric)
            trend_frames.append(tr)
            tr.to_csv(OUT/f'synthetic_trend_slopes_{metric}_{profile}.csv',index=False)
        pd.concat(trend_frames,ignore_index=True).to_csv(
            PLOT/f'synthetic_trend_slopes_{profile}.csv',index=False
        )

        # Plot-ready summaries; these are the exact numbers used by figure code.
        statsout.to_csv(PLOT/f'synthetic_effects_{profile}.csv',index=False)
        if valp.exists() and valp.stat().st_size>0:
            val=pd.read_csv(valp)
            for metric in ('auc','ap'):
                sign,sel,detail=validation_reliability(val,syn,metric)
                sign.to_csv(OUT/f'validation_sign_error_{metric}_{profile}.csv',index=False)
                sel.to_csv(OUT/f'validation_selection_accuracy_{metric}_{profile}.csv',index=False)
                detail.to_csv(OUT/f'validation_paired_detail_{metric}_{profile}.csv',index=False)
                sign.to_csv(PLOT/f'validation_sign_error_{metric}_{profile}.csv',index=False)
                sel.to_csv(PLOT/f'validation_selection_accuracy_{metric}_{profile}.csv',index=False)
    rnp=RESULTS/f'real_nested_{profile}.csv'
    if rnp.exists() and rnp.stat().st_size>0:
        rn=pd.read_csv(rnp); outs=[]
        for metric in METRICS:
            s,d=corrected_cv_summary(rn,metric,cfg); outs.append(s); d.to_csv(OUT/f'real_nested_paired_delta_{metric}_{profile}.csv',index=False)
        rout=pd.concat(outs,ignore_index=True); rout.to_csv(OUT/f'real_nested_statistics_{profile}.csv',index=False); rout.to_csv(PLOT/f'real_nested_effects_{profile}.csv',index=False)
    rsp=RESULTS/f'real_subsample_{profile}.csv'
    if rsp.exists() and rsp.stat().st_size>0:
        rs=pd.read_csv(rsp); outs=[]
        pair=['dataset','rep','m_train','pi_train','model']; groups=['dataset','m_train','pi_train','model','method']
        for metric in METRICS:
            s,d=summarize_delta(rs,pair,groups,metric,cfg,'real_subsample'); outs.append(s)
        out=pd.concat(outs,ignore_index=True); out.to_csv(OUT/f'real_subsample_statistics_{profile}.csv',index=False); out.to_csv(PLOT/f'real_subsample_effects_{profile}.csv',index=False)
    print(f'Analysis complete. Statistical tables: {OUT}. Plot-ready CSVs: {PLOT}.')
if __name__=='__main__': main()
