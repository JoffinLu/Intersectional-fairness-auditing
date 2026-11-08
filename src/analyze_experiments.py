"""Build journal figures/tables directly from retained experiment arrays."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom
from artifacts import ROOT, write_json, sha256, manifest
from audit_methods import fixed_prior_mse, fixed_prior_screen_power, weighted_reference

COLORS={'raw':'#525252','moment_eb':'#007C91','full_hb_holm':'#155B87',
        'split_conflict_holm':'#D36A29','split_raw_holm':'#699847','split_eb_holm':'#8064A2'}
LABELS={'raw':'Raw','moment_eb':'Moment EB','full_hb_holm':'Full HB + Holm',
        'split_conflict_holm':'Split conflict','split_raw_holm':'Split raw','split_eb_holm':'Split EB',
        'sr_point_adapter':'SR point adapter','suremap_author_adapter':'SureMap author code'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight',
 'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':160})

OUT=ROOT/'analysis'
FIG=ROOT/'paper'

def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf');fig.savefig(FIG/f'{name}.png',dpi=180);plt.close(fig)

def paired_flags(folder):
    records=[]
    for path in sorted(folder.glob('scenario_*.npz')):
        cfg=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        a=np.load(path);target=cfg['target_group'];full=a['flag__full_hb_holm'][:,target].astype(float)
        for method in ('split_uniform_holm','split_raw_holm','split_eb_holm','split_conflict_holm'):
            other=a[f'flag__{method}'][:,target].astype(float);d=other-full
            records.append({**cfg,'method':method,'full_power':float(full.mean()),'split_power':float(other.mean()),
                            'paired_difference':float(d.mean()),'paired_mcse':float(d.std(ddof=1)/np.sqrt(len(d))),
                            'repetitions':len(d)})
    return pd.DataFrame(records)

def baseline_details(folder):
    records=[]
    for path in sorted(folder.glob('baseline_*.npz')):
        cfg=json.loads(path.with_name(path.stem.replace('baseline','scenario')+'.json').read_text(encoding='utf-8'))
        a=np.load(path);p,n=a['p'],a['n'];target=cfg['target_group']
        estimates={k.split('__')[1]:a[k] for k in a.files if k.startswith('estimate__')}
        paired=np.logical_and.reduce([np.isfinite(x).all(axis=1) for x in estimates.values()])
        raw=estimates['raw'][paired];rawloss=((raw-p)**2).mean(axis=1)
        rawscreen=(raw-weighted_reference(raw,n)>.05)[:,target].astype(float)
        for method,allvalues in estimates.items():
            x=allvalues[paired];loss=((x-p)**2).mean(axis=1)
            screen=(x-weighted_reference(x,n)>.05)[:,target].astype(float)
            difference=loss-rawloss;sd=screen-rawscreen
            records.append({**cfg,'method':method,'attempted':len(allvalues),'paired_n':len(x),
              'failures':int((~np.isfinite(allvalues).all(axis=1)).sum()),'macro_mse':float(loss.mean()),
              'macro_mse_mcse':float(loss.std(ddof=1)/np.sqrt(len(x))),
              'target_mse':float(((x[:,target]-p[target])**2).mean()),
              'target_screen_rate':float(screen.mean()),'target_screen_mcse':float(screen.std(ddof=1)/np.sqrt(len(x))),
              'mse_difference_vs_raw':float(difference.mean()),'mse_difference_mcse':float(difference.std(ddof=1)/np.sqrt(len(x))),
              'screen_difference_vs_raw':float(sd.mean()),'screen_difference_mcse':float(sd.std(ddof=1)/np.sqrt(len(x)))})
    return pd.DataFrame(records)

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    folders={k:ROOT/'results'/v for k,v in {'main':'main_v1','ablation':'ablation_v1',
                  'semi':'semisynthetic_v2','baselines':'newer_baselines_v1'}.items()}
    frames={k:pd.read_csv(p/'summary.csv') for k,p in folders.items()}
    paired={k:paired_flags(p) for k,p in folders.items()}
    for k,df in paired.items():df.to_csv(OUT/f'{k}_paired_flags.csv',index=False)
    baselines=pd.concat([baseline_details(folders[k]).assign(source=k) for k in ('baselines','semi')],ignore_index=True)
    baselines.to_csv(OUT/'baseline_paired_details.csv',index=False)
    # Figure 1: exact mechanism, known fixed reference (separate from inference).
    kap=np.arange(0,151);normal=.1;harm=.35;n=10
    rawr,shrinkr=fixed_prior_mse(harm,n,normal,kap)
    raw0,shrink0=fixed_prior_mse(normal,n,normal,kap)
    macro=(4*shrink0+shrinkr)/5;rawmacro=float((4*raw0+rawr)/5)
    power=fixed_prior_screen_power(harm,n,normal,kap,.2)
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.65))
    ax[0].plot(kap,macro,label='Fixed-prior shrinkage',color=COLORS['moment_eb'])
    ax[0].axhline(rawmacro,color=COLORS['raw'],ls='--',label='Raw proportions')
    ax[0].set(xlabel='Prior concentration κ',ylabel='Macro-average MSE');ax[0].legend(frameon=False)
    ax[1].plot(kap,power,color=COLORS['moment_eb']);ax[1].axhline(power[0],color=COLORS['raw'],ls='--')
    ax[1].set(xlabel='Prior concentration κ',ylabel='Harmed-cell screen probability',ylim=(-.02,1.02))
    for a in ax:a.axvline(90,color='#888888',lw=.7,ls=':');a.grid(alpha=.15)
    fig.tight_layout();save(fig,'fig1_exact_mechanism')
    analytic=[]
    for strength in (0,1,5,20,90,100,500):
        r0,r1=fixed_prior_mse(.35,10,.1,strength)
        b0,b1=fixed_prior_mse(.1,10,.1,strength)
        analytic.append({'kappa':strength,'macro_mse_raw':float((4*b0+r0)/5),
                         'macro_mse_shrink':float((4*b1+r1)/5),
                         'harm_screen_probability':float(fixed_prior_screen_power(.35,10,.1,strength,.2))})
    pd.DataFrame(analytic).to_csv(OUT/'analytic_concentration_ablation.csv',index=False)
    # Figure 2: same-budget primary comparison, including negative results.
    df=frames['main'];s=df[(df.family=='sparse_rare')&(df.groups==32)]
    fig,axes=plt.subplots(1,3,figsize=(7.5,2.65),sharey=True)
    for ax,gap in zip(axes,(.10,.20,.30)):
        for method in ('full_hb_holm','split_conflict_holm','split_raw_holm','split_eb_holm'):
            z=s[(s.kind=='flag')&(s.method==method)&np.isclose(s.effect_above_margin,gap)].sort_values('target_n')
            ax.errorbar(z.target_n,z.target_flag_rate,yerr=1.96*z.target_mcse,marker='o',ms=3,lw=1,
                        color=COLORS[method],label=LABELS[method],capsize=2)
        ax.set(xscale='log',xlabel='Harmed-cell positive count n',title=f'Excess over margin = {gap:.2f}',ylim=(-.02,1.02))
        ax.set_xticks([10,20,40,80,160],labels=['10','20','40','80','160']);ax.grid(alpha=.15)
    axes[0].set_ylabel('Target flagging probability')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.3),ncol=4,frameon=False)
    fig.tight_layout();save(fig,'fig2_same_budget_power')
    # Figure 3: estimator decision diagnostic; no validity assertion.
    b=baselines[(baselines.source=='baselines')&(baselines.family=='sparse_rare')]
    selected=('raw','moment_eb','sr_point_adapter','suremap_author_adapter')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
    marker={'raw':'o','moment_eb':'s','sr_point_adapter':'^','suremap_author_adapter':'D'}
    palette=['#525252','#007C91','#A56427','#7255A3']
    scenario_order=list(b.scenario_id.drop_duplicates());xs=np.arange(len(scenario_order))
    for offset,(method,color) in enumerate(zip(selected,palette)):
        z=b[b.method==method].set_index('scenario_id').loc[scenario_order]
        pos=xs+(offset-1.5)*.16
        axes[0].errorbar(pos,z.macro_mse,yerr=1.96*z.macro_mse_mcse,fmt=marker[method],ms=4,capsize=2,color=color,label=LABELS[method])
        axes[1].errorbar(pos,z.target_screen_rate,yerr=1.96*z.target_screen_mcse,fmt=marker[method],ms=4,capsize=2,color=color)
    labels=[f'n={int(b[b.scenario_id==i].iloc[0].target_n)}\nε={b[b.scenario_id==i].iloc[0].effect_above_margin:.2f}' for i in scenario_order]
    for ax in axes:ax.set_xticks(xs,labels=labels);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Macro-average MSE');axes[1].set(ylabel='Harmed-cell threshold screen rate',ylim=(-.03,1.03))
    axes[0].legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(1,-.25))
    fig.tight_layout();save(fig,'fig3_newer_estimators')
    # Figure 4: split/floor ablation. Report paired difference relative to full.
    a=paired['ablation'];a=a[a.method=='split_conflict_holm']
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.5))
    for ax,nt in zip(axes,(20,80)):
        grid=a[a.target_n==nt].pivot(index='confirmation',columns='floor',values='paired_difference').sort_index()
        image=ax.imshow(grid,vmin=-.45,vmax=.45,cmap='BrBG',aspect='auto')
        ax.set_xticks(range(4),labels=['.05','.10','.50','1.0']);ax.set_yticks(range(3),labels=['.5','.8','.9'])
        ax.set(xlabel='Uniform weight floor ρ',ylabel='Confirmation fraction c',title=f'n = {nt}')
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{grid.iloc[i,j]:+.3f}',ha='center',va='center',fontsize=8)
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Power difference: split conflict − full',shrink=.85)
    save(fig,'fig4_split_ablation')
    # Figure 5: known-truth, real-support semi-synthetic comparisons.
    semi=paired['semi'];z=semi[(semi.method=='split_conflict_holm')].copy()
    fig,ax=plt.subplots(figsize=(7.4,3.4));x=np.arange(len(z))
    ax.errorbar(x,z.paired_difference,yerr=1.96*z.paired_mcse,fmt='o',ms=4,color=COLORS['split_conflict_holm'],capsize=2)
    ax.axhline(0,color='#555555',lw=.8)
    names=[f"{r.dataset.replace('acs_','ACS ')}\n{r.model.upper()} {'H' if r.geometry=='homogeneous' else 'V'}" for r in z.itertuples()]
    ax.set_xticks(x,labels=names,rotation=55,ha='right',fontsize=7)
    ax.set_ylabel('Power difference: split conflict − full');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();save(fig,'fig5_real_support')
    # Dataset and descriptive observed summaries (not population truth).
    datasets=[];observed=[]
    counts=pd.read_csv(ROOT/'data/processed/audit_group_counts.csv')
    for dataset in ('adult','diabetes','acs_income','acs_employment'):
        meta=json.loads((ROOT/f'data/processed/{dataset}_metadata.json').read_text(encoding='utf-8'))
        c=counts[counts.dataset==dataset];pos=c.audit_positive_n.to_numpy()
        datasets.append({'dataset':dataset,'rows':meta['rows'],'train':meta['split_rows']['train'],
          'calibration':meta['split_rows']['calibration'],'audit':meta['split_rows']['audit'],
          'positive_audit_n':int(pos.sum()),'registered_groups':len(pos),'nonzero_positive_groups':int((pos>0).sum()),
          'zero_positive_groups':int((pos==0).sum()),'positive_lt10_groups':int(((pos>0)&(pos<10)).sum()),
          'lr_audit_auc':meta['models']['lr']['audit_auc'],'hgb_audit_auc':meta['models']['hgb']['audit_auc']})
        for model in ('lr','hgb'):
            k=c[f'audit_{model}_fn'].to_numpy();r=np.divide(k,pos,out=np.full_like(k,np.nan,dtype=float),where=pos>0)
            for i,row in enumerate(c.itertuples()):
                observed.append({'dataset':dataset,'model':model,'groupkey':row.groupkey,'positive_n':int(pos[i]),
                  'fn':int(k[i]),'observed_fnr':None if pos[i]==0 else float(r[i]),
                  'observed_delta':None if pos[i]==0 else float(r[i]-(k.sum()-k[i])/(pos.sum()-pos[i]))})
    pd.DataFrame(datasets).to_csv(OUT/'dataset_table.csv',index=False)
    pd.DataFrame(observed).to_csv(OUT/'observed_groups.csv',index=False)
    # Machine-readable main headline evidence inventory.
    full=df[(df.kind=='flag')&(df.method=='full_hb_holm')]
    pair=paired['main'];pair=pair[(pair.method=='split_conflict_holm')&(pair.family=='sparse_rare')]
    flags=df[df.kind=='flag']
    metrics={'main_scenarios':int(df.scenario_id.nunique()),'main_repetitions':int(df.repetitions.iloc[0]),
      'max_empirical_fwer_all_valid':float(flags.fwer.max()),
      'max_empirical_fwer_full':float(full.fwer.max()),
      'sparse_conflict_vs_full':{'wins':int((pair.paired_difference>0).sum()),'ties':int((pair.paired_difference==0).sum()),
          'losses':int((pair.paired_difference<0).sum()),'total':len(pair),
          'min_difference':float(pair.paired_difference.min()),'max_difference':float(pair.paired_difference.max())},
      'analytic_example':{'raw_macro_mse':rawmacro,'shrink_macro_mse_k90':float(macro[90]),
                         'raw_screen_rate':float(power[0]),'shrink_screen_rate_k90':float(power[90])},
      'inputs':{str(p.relative_to(ROOT)):sha256(p) for folder in folders.values() for p in folder.glob('*.csv')}}
    write_json(OUT/'headline_metrics.json',metrics,version=True)
    # Wilson intervals remain informative for zero observed false flags.
    for field in ('fwer','target_flag_rate'):
        prob=flags[field].to_numpy();reps=flags.repetitions.to_numpy();zcrit=1.959963984540054
        center=(prob+zcrit*zcrit/(2*reps))/(1+zcrit*zcrit/reps)
        half=zcrit*np.sqrt(prob*(1-prob)/reps+zcrit*zcrit/(4*reps*reps))/(1+zcrit*zcrit/reps)
        flags=flags.assign(**{field+'_wilson_lower':np.maximum(0,center-half),field+'_wilson_upper':np.minimum(1,center+half)})
    flags.to_csv(OUT/'valid_flags_all_main.csv',index=False)
    frames['semi'].to_csv(OUT/'semisynthetic_summary.csv',index=False)
    print(json.dumps(metrics,indent=2))
    for path in OUT.glob('*.csv'):manifest(path,'Derived directly from retained counts and decisions',stage='analysis',skill='/analyze-results')

if __name__=='__main__':main()
