"""Regenerate interval and point-estimator presentation assets from retained outcomes."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from artifacts import ROOT, sha256, write_json
from make_tables import table


def wilson(rate, size=120):
    z=norm.ppf(.975)
    center=(rate+z*z/(2*size))/(1+z*z/size)
    half=z*np.sqrt(rate*(1-rate)/size+z*z/(4*size*size))/(1+z*z/size)
    return max(0., center-half), min(1., center+half)


def save(fig, name):
    fig.savefig(ROOT/'paper'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(ROOT/'paper'/f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    out=ROOT/'analysis/paper_assets';out.mkdir(exist_ok=True)
    controls=pd.read_csv(ROOT/'analysis/point_screen_controls.csv')
    methods={'raw':'Raw','moment_eb':'Moment EB','sr_point_adapter':'SR point','suremap_author_adapter':'SureMap'}
    sparse=controls[controls.family=='sparse_rare']
    rows=[]
    for sid in sparse.scenario_id.drop_duplicates():
        for method,label in methods.items():
            r=sparse[(sparse.scenario_id==sid)&(sparse.method==method)].iloc[0]
            lo,hi=wilson(r.target_screen_frequency)
            rows.append([f'{int(r.target_n)}/{r.effect_above_margin:.2f}',label,
                f'{1e3*r.macro_mse:.2f} ({1e3*r.macro_mse_mcse:.2f})',
                f'{1e3*r.target_mse:.2f} ({1e3*r.target_mse_mcse:.2f})',
                f'{100*r.target_screen_frequency:.1f} [{100*lo:.1f}, {100*hi:.1f}]'])
    table('table_estimators.tex','llrrr',[r'$n_*/\varepsilon$','Estimator',r'\shortstack{Macro MSE\\$\times10^3$}',
          r'\shortstack{Target MSE\\$\times10^3$}',r'\shortstack{Screen \%\\{[95\% MC interval]}}'],rows,
          r'Point-estimator diagnostics in four fixed sparse settings, 120 paired repetitions. Parentheses after MSE are Monte Carlo standard errors; brackets give pointwise 95\% Wilson intervals for simulation screen probabilities. Screen means estimated group--complement disparity exceeds .05, not a calibrated test. SR is the point adapter.','tab:estimators')
    intervals=pd.read_csv(ROOT/'analysis/interval_diagnostics/summary.csv')
    enriched=intervals.copy()
    metrics=['target_coverage','family_coverage','target_tolerance_rule','any_null_tolerance_rule','any_zero_null_rejection']
    for key in metrics:
        enriched[key+'_wilson_lower']=[wilson(p)[0] for p in enriched[key]]
        enriched[key+'_wilson_upper']=[wilson(p)[1] for p in enriched[key]]
    enriched.to_csv(out/'interval_summary_with_mc_intervals.csv',index=False)
    names={'wald_pointwise':'Wald-P','saft_style_pointwise':'SAFT-P','cp_simultaneous':'CP-S'}
    rows=[]
    for r in intervals.itertuples():
        rows.append([int(r.scenario_id[-3:])+1,names[r.method],
            f'{100*r.target_coverage:.1f}',f'{100*r.family_coverage:.1f}',
            f'{r.target_width:.3f}',
            f'{100*r.target_tolerance_rule:.1f} ({100*r.target_tolerance_rule_mcse:.1f})',
            f'{100*r.any_null_tolerance_rule:.1f} ({100*r.any_null_tolerance_rule_mcse:.1f})'])
    table('table_intervals.tex','rlrrrrr',['Setting','Interval',r'Cov-T\%',r'Cov-all\%','Width-T',r'Rule-T\%',r'Any-null\%'],rows,
          r'Exploratory interval diagnostics on the same 120 repetitions in each of the seven settings of Table~\ref{tab:pointcontrols}. Cov-T and Cov-all are target and whole-family contrast coverage; Width-T is mean target width. Rule-T and Any-null apply lower endpoint $>.05$, with MCSE in parentheses. Wald-P and SAFT-P use pointwise 95\% intervals; SAFT-P is the local conditional-FNR adaptation. CP-S uses a simultaneous 95\% exact-binomial rectangle. These unequal error targets do not give a matched-FWER power comparison. Rule-T is not power for nonharmful targets (settings 5 and 7). For 0/120 and 120/120, pointwise 95\% Wilson MC intervals are [0,3.1]\% and [96.9,100]\%; complete MC intervals are retained in CSV.','tab:intervals')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.titlesize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    colors={'full_hb_holm':'#155B87','split_conflict_holm':'#D36A29','split_raw_holm':'#699847','split_eb_holm':'#8064A2'}
    labels={'full_hb_holm':'Full HB','split_conflict_holm':'Split conflict','split_raw_holm':'Split raw','split_eb_holm':'Split EB'}
    main_data=pd.read_csv(ROOT/'results/main_v1/summary.csv')
    s=main_data[(main_data.family=='sparse_rare')&(main_data.groups==32)]
    fig,axes=plt.subplots(1,3,figsize=(6.1,3.25),sharey=True)
    for ax,gap in zip(axes,(.10,.20,.30)):
        for method in labels:
            z=s[(s.kind=='flag')&(s.method==method)&np.isclose(s.effect_above_margin,gap)].sort_values('target_n')
            ax.errorbar(z.target_n,z.target_flag_rate,yerr=1.96*z.target_mcse,marker='o',ms=3,lw=1,
                        color=colors[method],label=labels[method],capsize=2)
        ax.set(xscale='log',xlabel='Positive count',title=f'Excess = {gap:.2f}',ylim=(-.02,1.02))
        ax.set_xticks([10,20,40,80,160],labels=['10','20','40','80','160']);ax.tick_params(axis='x',labelsize=8)
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Target flag probability')
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));save(fig,'fig2_same_budget_power_ars')
    fig,axes=plt.subplots(1,2,figsize=(6.1,3.25))
    palette=['#525252','#007C91','#A56427','#7255A3'];markers=['o','s','^','D']
    order=list(sparse.scenario_id.drop_duplicates());x=np.arange(4)
    for offset,(method,label) in enumerate(methods.items()):
        z=sparse[sparse.method==method].set_index('scenario_id').loc[order]
        pos=x+(offset-1.5)*.16
        axes[0].errorbar(pos,z.macro_mse,yerr=1.96*z.macro_mse_mcse,fmt=markers[offset],ms=4,capsize=2,color=palette[offset],label=label)
        axes[1].errorbar(pos,z.target_screen_frequency,yerr=1.96*z.target_screen_mcse,fmt=markers[offset],ms=4,capsize=2,color=palette[offset])
    ticks=[f'n={int(sparse[sparse.scenario_id==sid].iloc[0].target_n)}\nε={sparse[sparse.scenario_id==sid].iloc[0].effect_above_margin:.2f}' for sid in order]
    for ax in axes:ax.set_xticks(x,labels=ticks);ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Macro-average MSE');axes[1].set(ylabel='Target screen probability',ylim=(-.03,1.03))
    handles,legends=axes[0].get_legend_handles_labels();fig.legend(handles,legends,loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));save(fig,'fig3_newer_estimators_ars')
    semi=pd.read_csv(ROOT/'analysis/semi_paired_flags.csv')
    z=semi[semi.method=='split_conflict_holm'].copy();x=np.arange(len(z))
    fig,axes=plt.subplots(2,1,figsize=(6.1,4.5),sharex=True)
    for field,label,color,marker,offset in [('full_power','Full HB','#155B87','o',-.08),('split_power','Split conflict','#D36A29','s',.08)]:
        probability=z[field].to_numpy();se=np.sqrt(probability*(1-probability)/(z.repetitions.to_numpy()-1))
        axes[0].errorbar(x+offset,100*probability,yerr=196*se,fmt=marker,ms=3,capsize=2,color=color,label=label)
    axes[0].set(ylabel='Target flagging (%)',ylim=(0,8));axes[0].legend(frameon=False,ncol=2,loc='upper left')
    axes[1].errorbar(x,100*z.paired_difference,yerr=196*z.paired_mcse,fmt='o',ms=4,color='#D36A29',capsize=2)
    axes[1].axhline(0,color='#555555',lw=.8);axes[1].set_ylabel('Split − full (pp)')
    short=[r.model.upper()+' '+('H' if r.geometry=='homogeneous' else 'V') for r in z.itertuples()]
    axes[1].set_xticks(x,labels=short,rotation=60,ha='right',fontsize=8)
    for start,label in zip(range(0,16,4),['Adult','Diabetes','ACS Income','ACS Employment']):
        axes[1].text(start+1.5,-.53,label,ha='center',va='top',transform=axes[1].get_xaxis_transform(),fontsize=9)
    for ax in axes:
        ax.grid(axis='y',alpha=.15)
        for split in (3.5,7.5,11.5):ax.axvline(split,color='#dddddd',lw=.6)
    fig.tight_layout(rect=(0,.09,1,1));save(fig,'fig5_real_support_ars')
    write_json(out/'asset_provenance.json',{'sources':{name:sha256(ROOT/name) for name in (
        'analysis/point_screen_controls.csv','analysis/semi_paired_flags.csv',
        'results/main_v1/summary.csv','analysis/interval_diagnostics/summary.csv')},
        'script_sha256':sha256(ROOT/'src/make_paper_assets.py'),
        'retained_original_figures':True,'numbers_changed':False,
        'changes':'New named readable figure exports; Figure5 adds absolute rates; Table2 adds target-MSE MCSE and Wilson screen intervals; new interval table.'})
    print('Three revised vector figures and Tables 2/9 generated from retained outputs.')


if __name__=='__main__':main()
