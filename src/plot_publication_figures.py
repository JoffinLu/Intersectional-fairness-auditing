"""Export all five figures from retained evidence; no experiment is rerun.

PDFs have a fixed 5.14-inch width to preserve >=9-point text at manuscript
scale. Editable SVG and 600-dpi preview PNG accompany each PDF. Data CSVs
retain every drawn ordinate and uncertainty endpoint.
"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import NullLocator, ScalarFormatter
from scipy.stats import norm
from artifacts import ROOT, sha256, write_json, now
from audit_methods import fixed_prior_mse, fixed_prior_screen_power
from plot_utils import WIDTH, PALETTE as C, style, clean, panel, ink_on

OUT = ROOT / 'analysis/figure_exports'
SOURCE_FILES = ('src/plot_publication_figures.py', 'src/plot_utils.py', 
    'src/audit_methods.py', 'analysis/point_screen_controls.csv',
    'analysis/ablation_paired_flags.csv', 'analysis/semi_paired_flags.csv',
    'results/main_v1/summary.csv')


def save(fig, number, data):
    name = f'Fig{number}'
    qa = OUT / 'qa'; qa.mkdir(exist_ok=True)
    colorbars = [ax for ax in fig.axes if getattr(ax, '_colorbar', None) is not None]
    exemptions = ([{'panels': ['b'], 'checks': ['panel-width'],
        'reason': 'Existing absolute-power panel is wider; the two panels have different x metrics.'}]
        if number == 5 else [])
    # A fixed canvas, rather than a tight bbox, keeps font scaling predictable.
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png', dpi=600)
    shutil.copy2(OUT/f'{name}.pdf', ROOT/'paper'/f'{name}.pdf')
    pd.DataFrame(data).to_csv(OUT / f'{name}_data.csv', index=False)
    plt.close(fig)


def figure1():
    k = np.arange(151)
    r0, r1 = fixed_prior_mse(.35, 10, .1, k)
    b0, b1 = fixed_prior_mse(.1, 10, .1, k)
    raw = float((4*b0+r0)/5)
    risk = (4*b1+r1)/5
    screen = fixed_prior_screen_power(.35, 10, .1, k, .2)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.05))
    fig.subplots_adjust(left=.135, right=.985, bottom=.21, top=.76, wspace=.39)
    for ax in axes:
        ax.set(xlim=(-3, 153), xlabel='Prior concentration κ')
        ax.set_xticks([0, 50, 100, 150])
        ax.axvline(90, color='#8C8C8C', ls=':', lw=1)
        clean(ax)
    axes[0].axhline(1000*raw, color=C['raw'], ls='--', label='Raw proportions')
    axes[0].plot(k, 1000*risk, color=C['shrink'], label='Fixed-prior shrinkage')
    axes[0].set(ylabel='Macro MSE × 1,000', ylim=(5.5, 12.5), yticks=[6, 8, 10, 12])
    axes[0].plot(90, 1000*risk[90], 'o', color=C['shrink'], ms=4)
    axes[0].annotate('κ = 90: 10.24', xy=(90, 1000*risk[90]), xytext=(5, 10.8),
                     fontsize=9.5, arrowprops={'arrowstyle':'->', 'lw':.7, 'color':C['raw']})
    axes[1].axhline(100*screen[0], color=C['raw'], ls='--')
    axes[1].plot(k, 100*screen, color=C['shrink'])
    axes[1].plot(90, 0, 'o', color=C['shrink'], ms=4, clip_on=False)
    axes[1].set(ylabel='Target screen probability (%)', ylim=(-3, 103), yticks=[0, 25, 50, 75, 100])
    axes[1].annotate('κ = 90: 0%', xy=(90, 0), xytext=(98, 35), fontsize=9.5,
                     arrowprops={'arrowstyle':'->', 'lw':.7, 'color':C['raw']})
    panel(axes[0], '(a) Average estimation risk')
    panel(axes[1], '(b) Harmed-cell screening')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53,.975), ncol=2,
               frameon=False, handlelength=2.5, columnspacing=1.4)
    save(fig, 1, {'kappa':k, 'raw_macro_mse':raw, 'shrink_macro_mse':risk,
                  'raw_target_screen_probability':screen[0], 'shrink_target_screen_probability':screen})


def figure2():
    source = pd.read_csv(ROOT / 'analysis/point_screen_controls.csv')
    s = source[source.family == 'sparse_rare']
    order = list(s.scenario_id.drop_duplicates())
    methods = [('raw','Raw',C['raw'],'o'), ('moment_eb','Moment EB',C['eb'],'s'),
               ('sr_point_adapter','SR point',C['sr'],'^'), ('suremap_author_adapter','SureMap',C['suremap'],'D')]
    fig, axes = plt.subplots(1,2,figsize=(WIDTH,3.55))
    fig.subplots_adjust(left=.13, right=.985, bottom=.28, top=.81, wspace=.4)
    records=[]
    for j,(method,label,color,marker) in enumerate(methods):
        z=s[s.method==method].set_index('scenario_id').loc[order]
        x=np.arange(4)+(j-1.5)*.16
        axes[0].errorbar(x,1000*z.macro_mse,yerr=1960*z.macro_mse_mcse,
                         fmt=marker,ms=4.2,capsize=2,color=color,label=label,elinewidth=.9)
        p=z.target_screen_frequency.to_numpy();n=z.repetitions.to_numpy();q=norm.ppf(.975)
        center=(p+q*q/(2*n))/(1+q*q/n)
        half=q*np.sqrt(p*(1-p)/n+q*q/(4*n*n))/(1+q*q/n)
        lo=np.maximum(0,center-half);hi=np.minimum(1,center+half)
        axes[1].errorbar(x,100*p,yerr=100*np.array([p-lo,hi-p]),fmt=marker,ms=4.2,
                         capsize=2,color=color,elinewidth=.9)
        for idx,(sid,row) in enumerate(z.iterrows()):
            records.append({'scenario_id':sid,'method':method,'x':x[idx],
                'macro_mse':row.macro_mse,'macro_mse_mcse':row.macro_mse_mcse,
                'target_screen_probability':p[idx],'screen_wilson_lower':lo[idx],
                'screen_wilson_upper':hi[idx],'repetitions':int(n[idx])})
    ticks=[f'{int(s[s.scenario_id==sid].iloc[0].target_n)}\n{s[s.scenario_id==sid].iloc[0].effect_above_margin:.2f}' for sid in order]
    for ax in axes:
        ax.set_xticks(range(4),labels=ticks)
        ax.set(xlim=(-.5,3.5),xlabel='Target count n / excess ε')
        clean(ax)
    axes[0].set(ylabel='Macro MSE × 1,000',ylim=(0,4.4),yticks=[0,1,2,3,4])
    axes[1].set(ylabel='Target screen probability (%)',ylim=(-3,103),yticks=[0,25,50,75,100])
    panel(axes[0], '(a) Macro risk');panel(axes[1], '(b) Target screening')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.99),ncol=4,
               frameon=False,columnspacing=.8,handletextpad=.4)
    save(fig,2,records)


def figure3():
    s=pd.read_csv(ROOT/'results/main_v1/summary.csv')
    s=s[(s.family=='sparse_rare')&(s.groups==32)&(s.kind=='flag')]
    methods=[('full_hb_holm','Full HB + Holm',C['full'],'o','-'),
             ('split_conflict_holm','Split conflict',C['conflict'],'s','--'),
             ('split_raw_holm','Split raw',C['split_raw'],'^','-.'),
             ('split_eb_holm','Split EB',C['split_eb'],'D',':')]
    fig,axes=plt.subplots(1,3,figsize=(WIDTH,3.4),sharey=True)
    fig.subplots_adjust(left=.12,right=.985,bottom=.36,top=.85,wspace=.2)
    records=[]
    for ax,gap,letter in zip(axes,(.10,.20,.30),'abc'):
        for method,label,color,marker,ls in methods:
            z=s[(s.method==method)&np.isclose(s.effect_above_margin,gap)].sort_values('target_n')
            ax.errorbar(z.target_n,100*z.target_flag_rate,yerr=196*z.target_mcse,
                marker=marker,ls=ls,ms=3.5,lw=1.25,capsize=1.5,color=color,label=label,elinewidth=.7)
            records.extend(z[['scenario_id','target_n','effect_above_margin','method','target_flag_rate','target_mcse','repetitions']].to_dict('records'))
        ax.set(xscale='log',ylim=(-3,103),xlim=(8.5,188),yticks=[0,25,50,75,100])
        ax.set_xticks([10,20,40,80,160],labels=['10','20','40','80','160'])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='x',labelsize=9)
        panel(ax,f'({letter}) ε = {gap:.2f}');clean(ax)
    axes[0].set_ylabel('Target flagging probability (%)')
    fig.text(.55,.22,'Target positive count n (log scale)',ha='center',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,
               frameon=False,columnspacing=1.8,handlelength=2.7)
    save(fig,3,records)


def figure4():
    s=pd.read_csv(ROOT/'analysis/ablation_paired_flags.csv')
    s=s[s.method=='split_conflict_holm']
    # Navy-coral zero-centred diverging scale; signed cell values are a
    # redundant non-colour encoding. No cell values are saturated.
    cmap=LinearSegmentedColormap.from_list('coral_white_navy',
        [C['heat_negative'],C['heat_zero'],C['heat_positive']])
    fig=plt.figure(figsize=(WIDTH,3.05))
    gs=fig.add_gridspec(1,3,width_ratios=[1,1,.055],left=.15,right=.93,bottom=.30,top=.82,wspace=.19)
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
    records=[]
    for ax,n,letter in zip(axes,(20,80),'ab'):
        grid=s[s.target_n==n].pivot(index='confirmation',columns='floor',values='paired_difference').sort_index()
        image=ax.pcolormesh(np.arange(5)-.5,np.arange(4)-.5,100*grid.to_numpy(),
            cmap=cmap,norm=TwoSlopeNorm(vmin=-40,vcenter=0,vmax=40),rasterized=False)
        ax.set(xlim=(-.5,3.5),ylim=(2.5,-.5))
        ax.set_xticks(range(4),labels=['.05','.10','.50','1.0'])
        ax.set_yticks(range(3),labels=['.5','.8','.9'])
        ax.tick_params(length=0,pad=5)
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.0);ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
        panel(ax,f'({letter}) Target count n = {n}')
        for i,c in enumerate(grid.index):
            for j,rho in enumerate(grid.columns):
                value=100*grid.iloc[i,j]
                ax.text(j,i,f'{value:+.1f}',ha='center',va='center',fontsize=9.5,
                        color=ink_on(image.cmap(image.norm(value))))
                row=s[(s.target_n==n)&np.isclose(s.confirmation,c)&np.isclose(s.floor,rho)].iloc[0]
                records.append({'target_n':n,'confirmation':c,'floor':rho,
                    'paired_difference':row.paired_difference,'paired_mcse':row.paired_mcse,
                    'displayed_difference_pp':value,'full_power':row.full_power,'split_power':row.split_power})
    axes[0].set_ylabel('Confirmation fraction c')
    axes[1].tick_params(labelleft=False)
    cb=fig.colorbar(image,cax=fig.add_subplot(gs[0,2]),ticks=[-40,-20,0,20,40])
    cb.solids.set_rasterized(False)
    cb.ax.tick_params(labelsize=9);cb.outline.set_visible(False)
    fig.text(.51,.12,'Uniform weight floor ρ',ha='center',fontsize=10)
    fig.text(.51,.025,'Paired power difference: split conflict − full (pp)',ha='center',fontsize=9.5)
    save(fig,4,records)


def figure5():
    s=pd.read_csv(ROOT/'analysis/semi_paired_flags.csv')
    z=s[s.method=='split_conflict_holm'].copy().reset_index(drop=True)
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,4.95),sharey=True,gridspec_kw={'width_ratios':[1.18,1]})
    fig.subplots_adjust(left=.24,right=.985,bottom=.14,top=.84,wspace=.23)
    labels=[];ticks=[];records=[]
    datasets=['Adult','Diabetes','ACS Income','ACS Employment']
    for group,dataset in enumerate(datasets):
        ticks.append(group*5);labels.append(dataset)
        for j in range(4):
            idx=group*4+j;row=z.iloc[idx];y=group*5+j+1
            ticks.append(y);labels.append(row.model.upper()+' · '+('H' if row.geometry=='homogeneous' else 'V'))
            for field,label,color,marker,offset in [('full_power','Full HB',C['full'],'o',-.12),
                ('split_power','Split conflict',C['conflict'],'s',.12)]:
                p=float(row[field]);se=np.sqrt(p*(1-p)/(row.repetitions-1))
                axes[0].errorbar(100*p,y+offset,xerr=196*se,fmt=marker,ms=3.6,
                    color=color,capsize=2,elinewidth=.8,label=label if idx==0 else None)
            axes[1].errorbar(100*row.paired_difference,y,xerr=196*row.paired_mcse,
                fmt='D',ms=3.6,color=C['difference'],capsize=2,elinewidth=.8)
            records.append({**row.to_dict(),'y':y,
                'full_mcse':np.sqrt(row.full_power*(1-row.full_power)/(row.repetitions-1)),
                'split_mcse':np.sqrt(row.split_power*(1-row.split_power)/(row.repetitions-1))})
    for ax in axes:
        ax.set(ylim=(19.8,-.75));clean(ax,'x')
        ax.tick_params(axis='y',length=0,pad=6)
        for boundary in (4.6,9.6,14.6):
            ax.axhline(boundary,color='#D9DDE0',lw=.6)
    axes[0].set_yticks(ticks,labels=labels)
    for text in axes[0].get_yticklabels():
        if text.get_text() in datasets:text.set_fontweight('bold')
    axes[1].tick_params(labelleft=False)
    axes[0].set(xlim=(0,8),xticks=[0,2,4,6,8],xlabel='Target flagging (%)')
    axes[1].set(xlim=(-1,5),xticks=[0,2,4],xlabel='Split − full (pp)')
    axes[1].axvline(0,color='#777777',lw=.8,ls='--')
    panel(axes[0],'(a) Absolute target power');panel(axes[1],'(b) Paired gain')
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='upper center',bbox_to_anchor=(.57,.965),ncol=2,
               frameon=False,columnspacing=2)
    save(fig,5,records)


def main():
    OUT.mkdir(exist_ok=True,parents=True);(ROOT/'paper').mkdir(exist_ok=True);style()
    # These are enforced here as well as in the shared style helper so that
    # this entry point states its editable-vector export contract explicitly.
    plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42})
    for function in (figure1,figure2,figure3,figure4,figure5):function()
    import fitz
    exports={}
    for i in range(1,6):
        path=ROOT/f'paper/Fig{i}.pdf'
        with fitz.open(path) as doc:
            spans=[s for b in doc[0].get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
            exports[path.name]={'sha256':sha256(path),'width_pt':doc[0].rect.width,
                'height_pt':doc[0].rect.height,'min_export_text_pt':min(s['size'] for s in spans),
                'raster_image_objects':len(doc[0].get_images()),'vector_drawing_objects':len(doc[0].get_drawings())}
    write_json(OUT/'provenance.json',{'generated_at':now(),'no_new_experiments':True,
        'source_hashes':{p:sha256(ROOT/p) for p in SOURCE_FILES},'exports':exports,
        'data_exports':{f'Fig{i}_data.csv':sha256(OUT/f'Fig{i}_data.csv') for i in range(1,6)},
        'uncertainty_change':'Figure2 screen bars now use bounded pointwise Wilson 95% MC intervals, the same calculation as Table2; risk bars and all other error bars remain 1.96 MCSE.',
        'palette':C,'palette_reference':'https://nanx.me/ggsci/reference/pal_npg.html',
        'style':'Unified NPG-inspired navy/coral/teal/slate/cyan/graphite; Arial; fixed 5.14-inch canvas; min 9-point text; redundant line/marker/signed-value encodings; zero-centred heatmap in pp.'})
    print(json.dumps(exports,indent=2))


if __name__=='__main__':main()
