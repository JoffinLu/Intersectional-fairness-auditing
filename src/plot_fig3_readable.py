"""Redraw Fig. 3 from the same retained outcomes, with explicit paired contrasts."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.lines import Line2D
from artifacts import ROOT, sha256, write_json, now
from plot_utils import WIDTH, PALETTE, style, clean, panel

OUT=ROOT/'analysis/figure3_exports'
METHODS=[('full_hb_holm','Full HB + Holm','full','o','-'),
         ('split_conflict_holm','Split conflict','conflict','s','--'),
         ('split_raw_holm','Split raw','split_raw','^','-.'),
         ('split_eb_holm','Split EB','split_eb','D',':')]

def collect():
    source=ROOT/'results/main_v1/summary.csv'
    summary=pd.read_csv(source)
    selected=summary[(summary.family=='sparse_rare')&(summary.groups==32)&
                     (summary.kind=='flag')&summary.method.isin([m[0] for m in METHODS])]
    assert len(selected)==60 and selected.scenario_id.nunique()==15
    records=[]; inputs={'results/main_v1/summary.csv':sha256(source)}
    for sid in sorted(selected.scenario_id.unique()):
        raw_path=ROOT/f'results/main_v1/{sid}.npz'
        config_path=ROOT/f'results/main_v1/{sid}.json'
        config=json.loads(config_path.read_text(encoding='utf-8'))
        target=config['target_group']
        for path in (raw_path,config_path):inputs[path.relative_to(ROOT).as_posix()]=sha256(path)
        with np.load(raw_path,allow_pickle=False) as raw:
            baseline=raw['flag__full_hb_holm'][:,target].astype(float)
            assert len(baseline)==2000
            for j,(method,label,key,marker,ls) in enumerate(METHODS):
                row=selected[(selected.scenario_id==sid)&(selected.method==method)].iloc[0]
                flags=raw['flag__'+method][:,target].astype(float)
                mean=float(flags.mean());se=float(flags.std(ddof=1)/np.sqrt(len(flags)))
                assert np.isclose(mean,row.target_flag_rate,rtol=0,atol=1e-14)
                assert np.isclose(se,row.target_mcse,rtol=0,atol=1e-14)
                difference=flags-baseline
                gain=float(difference.mean());paired_se=float(difference.std(ddof=1)/np.sqrt(len(flags)))
                records.append({'scenario_id':sid,'target_n':int(row.target_n),
                    'effect_above_margin':float(row.effect_above_margin),'method':method,
                    'target_flag_rate':mean,'target_mcse':se,'repetitions':len(flags),
                    'absolute_lower_pct':100*(mean-1.96*se),'absolute_upper_pct':100*(mean+1.96*se),
                    'paired_difference_pp':100*gain,'paired_mcse_pp':100*paired_se,
                    'paired_lower_pp':100*(gain-1.96*paired_se),'paired_upper_pp':100*(gain+1.96*paired_se),
                    'log2_absolute_display_offset':(j-1.5)*.20,
                    'log2_contrast_display_offset':(j-2)*.24 if j else 0})
    return pd.DataFrame(records),inputs

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'qa').mkdir(exist_ok=True)
    style()
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42})
    data,inputs=collect()
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,4.95),sharex=True,sharey='row',
                          gridspec_kw={'height_ratios':[1.25,1]})
    fig.subplots_adjust(left=.135,right=.985,bottom=.175,top=.82,wspace=.23,hspace=.36)
    for col,gap in enumerate((.10,.20,.30)):
        top,bottom=axes[:,col]
        for j,(method,label,key,marker,ls) in enumerate(METHODS):
            z=data[(data.method==method)&np.isclose(data.effect_above_margin,gap)].sort_values('target_n')
            assert z.target_n.tolist()==[10,20,40,80,160]
            color=PALETTE[key]
            x=z.target_n.to_numpy()*2**z.log2_absolute_display_offset.to_numpy()
            top.errorbar(x,100*z.target_flag_rate,yerr=196*z.target_mcse,
                marker=marker,ls='none',ms=3.2,capsize=1.1,color=color,
                markerfacecolor='white' if j==0 else color,markeredgewidth=.85,
                elinewidth=.75,capthick=.75,zorder=5-j)
            if j:
                x=z.target_n.to_numpy()*2**z.log2_contrast_display_offset.to_numpy()
                bottom.errorbar(x,z.paired_difference_pp,yerr=1.96*z.paired_mcse_pp,
                    fmt=marker,ms=3.2,color=color,capsize=1.1,elinewidth=.75,
                    capthick=.75,markeredgewidth=.7,zorder=5-j)
        for ax in (top,bottom):
            ax.set(xscale='log',xlim=(7.3,218))
            ax.set_xticks([10,20,40,80,160],labels=['10','20','40','80','160'])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='x',labelsize=9,pad=3)
            clean(ax)
        top.set(ylim=(-3,103),yticks=[0,25,50,75,100])
        bottom.set(ylim=(-8.5,6.5),yticks=[-8,-4,0,4])
        bottom.axhline(0,color=PALETTE['full'],lw=.9,zorder=2)
        panel(top,f'({"abc"[col]}) ε = {gap:.2f}')
        panel(bottom,f'({"def"[col]}) ε = {gap:.2f}')
    axes[0,0].set_ylabel('Target flagging (%)',labelpad=5)
    axes[1,0].set_ylabel('Split − full (pp)',labelpad=5)
    handles=[Line2D([],[],color=PALETTE[key],marker=marker,ls='none',ms=3.8,
                        markerfacecolor='white' if j==0 else PALETTE[key],label=label)
             for j,(_,label,key,marker,ls) in enumerate(METHODS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.985),ncol=2,
               frameon=False,columnspacing=1.5,handlelength=2.3,handletextpad=.6)
    fig.text(.56,.085,'Target positive count n (log scale)',ha='center',fontsize=10)
    fig.text(.56,.025,'Within-count marker offsets are for visibility only',ha='center',fontsize=9)
    for suffix in ('pdf','svg','png'):
        fig.savefig(OUT/f'Fig3.{suffix}',**({'dpi':600} if suffix=='png' else {}))
    fig.savefig(OUT/'Fig3_preview.png',dpi=300)
    plt.close(fig)
    data.to_csv(OUT/'Fig3_data.csv',index=False)
    inputs.update({name:sha256(ROOT/name) for name in ('src/plot_fig3_readable.py','src/plot_utils.py','src/artifacts.py')})
    write_json(OUT/'provenance.json',{'generated_at':now(),'new_experiments':False,
        'source_sha256':inputs,'absolute_rows':60,'paired_nonbaseline_rows':45,
        'all_2000_paired_replicates_retained':True,'paired_mcse':'std(flag_split - flag_full, ddof=1) / sqrt(2000)',
        'intervals':'mean +/-1.96 MCSE; pointwise Monte Carlo uncertainty, not simultaneous intervals',
        'axis_offsets':'x_display = nominal_count * 2**method_offset; ordinate and uncertainty unchanged',
        'palette':PALETTE,'outputs':{f'Fig3.{s}':sha256(OUT/f'Fig3.{s}') for s in ('pdf','svg','png')},
        'data_sha256':sha256(OUT/'Fig3_data.csv')})
    shutil.copy2(OUT/'Fig3.pdf',ROOT/'paper/Fig3.pdf')
    print('Fig3: retained60 absolute points; added45 paired contrasts from existing outcomes.')

if __name__=='__main__':main()
