"""Publication figures from frozen tables/predictions. No estimator fitting."""

from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parents[1]/'publication'

H=['india','malaysia_safe','taiwan','usa']

N={'india':'India','malaysia_safe':'Malaysia SAFE','taiwan':'Taiwan','usa':'USA'}

M=['training_mean','training_median','ssi_raw_euclidean','ssi_isotonic','gdm','pairwise_random_forest_ssi','siamese_ensemble_ssi']

L=dict(zip(M,['Training mean','Training median','Raw SSI distance','Isotonic SSI','GDM','Pairwise forest','Siamese ensemble']))

C=dict(zip(M,['#666666','#999999','#CC79A7','#E69F00','#009E73','#0072B2','#D55E00']))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'savefig.dpi':600})

FIG=None

FIG=None

T=OUT/'tables'

def read(path):return pd.read_csv(path,float_precision='round_trip',low_memory=False)

def save(fig,name):
    for ext in ['pdf','png']:
        fig.savefig(FIG/f'{name}.{ext}',bbox_inches='tight',pad_inches=.06,metadata={'Creator':'Reproducible manuscript plotting script'} if ext=='pdf' else None)
    plt.close(fig)

def panel(ax,label):ax.text(-.13,1.06,label,transform=ax.transAxes,weight='bold',fontsize=10)

def target_distributions(root,pred):
    fig,axs=plt.subplots(1,4,figsize=(7.16,2.35),sharex=True,sharey=True)
    for i,h in enumerate(H):
        train=read(root/f'experiments/holdout_{h}/filtered/outer_pairs.csv')
        vals=train.loc[train.pair_role=='final_train','jaccard_distance'].to_numpy()
        y=pred.loc[(pred.outer_holdout==h)&(pred.data_condition=='filtered')&(pred.pair_role=='unseen_test'),'true_jaccard_distance'].to_numpy()
        ax=axs[i]
        ax.hist(vals,bins=np.linspace(0,1,26),weights=np.ones(len(vals))/len(vals),histtype='step',lw=1.2,color='#0072B2',label='Pooled source training')
        ax.hist(y,bins=np.linspace(0,1,26),weights=np.ones(len(y))/len(y),histtype='step',lw=1.2,ls='--',color='#D55E00',label='Withheld test')
        ax.axvline(vals.mean(),lw=.8,ls=':',color='#0072B2');ax.axvline(y.mean(),lw=.8,ls=':',color='#D55E00')
        ax.set_title(N[h]);ax.set_xlabel('Observed Jaccard distance');ax.set_xlim(0,1);panel(ax,chr(97+i))
    axs[0].set_ylabel('Fraction of pairs per bin')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,-.1),frameon=False)
    fig.tight_layout(w_pad=.8);save(fig,'fig02_target_distributions')

def performance(ci,metric,cond,name):
    methods=[m for m in M if (m!='ssi_raw_euclidean' if metric=='mae' else m not in ['training_mean','training_median'])]
    fig,axs=plt.subplots(2,4,figsize=(7.16,4.6),sharex=True,sharey=True)
    for r,role in enumerate(['familiar_test','unseen_test']):
        for c,h in enumerate(H):
            ax=axs[r,c];panel(ax,chr(97+4*r+c))
            f=ci.loc[(ci.outer_holdout==h)&(ci.data_condition==cond)&(ci.pair_role==role)&(ci.metric==metric)]
            f=f.loc[f.dataset==('familiar_macro' if r==0 else h)].set_index('method')
            for y,m in enumerate(methods):
                row=f.loc[m];v=row.estimate
                if np.isfinite(v):
                    ax.plot([row.ci_lower,row.ci_upper],[y,y],lw=1,color=C[m]);ax.plot(v,y,'o',ms=4,color=C[m])
                else:ax.text(.03,y,'undefined',fontsize=6,va='center')
            ax.set_yticks(range(len(methods)),[L[m] for m in methods])
            ax.grid(axis='x',color='#DDDDDD',lw=.5)
            ax.set_title(N[h] if r==0 else '')
            ax.set_xlabel('MAE' if metric=='mae' else 'Spearman correlation')
            if c==0:ax.set_ylabel('Familiar macro' if r==0 else 'Withheld dataset',fontweight='bold')
            if metric=='spearman':ax.axvline(0,color='black',lw=.6);ax.axvline(.2,color='#999999',lw=.6,ls=':')
    axs[0,0].invert_yaxis()
    if metric=='mae':axs[0,0].set_xlim(.065,.36)
    else:axs[0,0].set_xlim(-.25,.50)
    shown=ci.loc[(ci.data_condition==cond)&(ci.metric==metric)&ci.method.isin(methods)&((ci.pair_role=='unseen_test')|(ci.dataset=='familiar_macro'))]
    lo,hi=axs[0,0].get_xlim()
    assert shown.ci_lower.min()>=lo and shown.ci_upper.max()<=hi, 'Do not truncate an uncertainty interval'
    fig.tight_layout(w_pad=.6,h_pad=2);save(fig,name)

def calibration(pred,condition,name):
    methods=['ssi_isotonic','gdm','pairwise_random_forest_ssi','siamese_ensemble_ssi']
    fig,axs=plt.subplots(4,4,figsize=(7.16,6.9),sharex=True,sharey=True)
    for r,m in enumerate(methods):
        for c,h in enumerate(H):
            g=pred.loc[(pred.outer_holdout==h)&(pred.data_condition==condition)&(pred.pair_role=='unseen_test')];ax=axs[r,c]
            mesh=ax.hexbin(g[m],g.true_jaccard_distance,gridsize=24,extent=(0,1,0,1),mincnt=1,bins='log',cmap='Blues',rasterized=True,vmin=1,vmax=1000)
            ax.plot([0,1],[0,1],ls='--',color='#D55E00',lw=.8);ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
            if r==0:ax.set_title(N[h])
            if c==0:ax.set_ylabel(L[m]+'\nObserved Jaccard')
            if r==3:ax.set_xlabel('Predicted Jaccard')
            ax.text(.03,.94,chr(97+r*4+c),transform=ax.transAxes,weight='bold',va='top')
    fig.tight_layout(w_pad=.6,h_pad=.6,rect=(0,.065,1,1))
    cax=fig.add_axes([.36,.027,.30,.013])
    bar=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[1,10,100,1000],extend='max')
    bar.set_label('Pairs per hexagon (log scale)',fontsize=7)
    bar.ax.tick_params(labelsize=6)
    save(fig,name)

def shift_compression(metrics,failure):
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.75));panel(axs[0],'a');panel(axs[1],'b')
    f=failure.loc[(failure.data_condition=='filtered')&(failure.pair_role=='unseen_test')&(failure.metric=='ssi_ks_statistic')]
    for k,h in enumerate(H):
        vals=f.loc[f.outer_holdout==h,'estimate'].to_numpy()
        offsets=np.linspace(-.17,.17,len(vals))
        axs[0].scatter(np.full(len(vals),k)+offsets,np.sort(vals),s=6,alpha=.65,color='#0072B2')
        axs[0].plot([k-.2,k+.2],[np.median(vals)]*2,color='black',lw=1.3)
    axs[0].set_xticks(range(4),[N[h].replace('Malaysia SAFE','Malaysia') for h in H]);axs[0].set(ylim=(0,1.04),ylabel='Raw SSI distribution difference (KS)')
    for z,m in enumerate(['ssi_isotonic','gdm','pairwise_random_forest_ssi','siamese_ensemble_ssi']):
        f=metrics.loc[(metrics.data_condition=='filtered')&(metrics.pair_role=='unseen_test')&(metrics.metric=='prediction_sd_divided_by_target_sd')&(metrics.method==m)].set_index('outer_holdout')
        axs[1].plot(np.arange(4)+(z-1.5)*.12,[f.loc[h,'estimate'] for h in H],['o','s','^','D'][z],ms=4,color=C[m],label=L[m])
    axs[1].axhline(1,color='#888888',ls='--',lw=.8);axs[1].axhline(.5,color='#AAAAAA',ls=':',lw=.8)
    axs[1].set_xticks(range(4),[N[h].replace('Malaysia SAFE','Malaysia') for h in H]);axs[1].set(ylim=(0,1.1),ylabel='Prediction SD / target SD')
    fig.legend(*axs[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.13),ncol=4,frameon=False,fontsize=7)
    fig.tight_layout();save(fig,'fig06_shift_compression')

def lf_and_preproc(root):
    f=read(T/'lf_representation_statistics.csv');f=f.loc[f.level=='source_frame']
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.6))
    colors=['#0072B2','#009E73','#D55E00','#CC79A7']
    for h,col,style in zip(H,colors,['-','--',':','-.']):
        g=f.loc[f.dataset==h].sort_values('lf_dimension')
        axs[0].plot(range(128),100*g.proportion_negative,lw=.9,color=col,ls=style,label=N[h]);axs[1].plot(range(128),100*g.proportion_exactly_zero,lw=.9,color=col,ls=style)
    for ax in axs:ax.set(xlabel='Stored LF coordinate',ylim=(-2,102),xlim=(0,127))
    axs[0].set_ylabel('Negative values (%)');axs[1].set_ylabel('Exact zero values (%)');panel(axs[0],'a');panel(axs[1],'b')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.1),ncol=4,frameon=False);fig.tight_layout();save(fig,'supp01_lf_fingerprints')
    hist=read(root/'results/ssi_preprocessing_comparison/epoch_history.csv')
    sel=read(root/'results/ssi_preprocessing_comparison/selected_preprocessing.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.16,2.7));panel(axs[0],'a');panel(axs[1],'b')
    for i,method in enumerate(['STANDARD','ASINH_STANDARD']):
        g=hist.loc[hist.preprocessing_method==method].copy();g['extreme']=g[['logit_minimum','logit_maximum']].abs().max(axis=1)
        vals=g.groupby('fit_id').extreme.max().to_numpy()
        axs[0].scatter(i+np.linspace(-.18,.18,len(vals)),np.sort(vals),s=7,color=['#0072B2','#D55E00'][i],alpha=.6)
    axs[0].set_xticks([0,1],['Standard','Asinh + standard']);axs[0].set(yscale='log',ylabel='Maximum absolute logit over each fit')
    for c,cond in enumerate(['filtered','author_compatible']):
        for k,h in enumerate(H):
            g=sel.loc[(sel.outer_holdout==h)&(sel.data_condition==cond)].set_index('preprocessing_method')
            d=g.loc['ASINH_STANDARD','mean_validation_mae']-g.loc['STANDARD','mean_validation_mae']
            axs[1].plot(k+(-.12 if c==0 else .12),d,'o' if c==0 else 's',color='#0072B2' if c==0 else '#999999',label=cond if k==0 else '')
    axs[1].axhline(0,lw=.7,color='black');axs[1].set_xticks(range(4),['India','Malaysia','Taiwan','USA']);axs[1].set_ylabel('Inner MAE: asinh minus standard')
    axs[1].legend([Line2D([],[],marker='o',color='#0072B2',ls=''),Line2D([],[],marker='s',color='#999999',ls='')],['Filtered','Author-compatible'],frameon=False,fontsize=7)
    fig.tight_layout();save(fig,'supp02_preprocessing')

def architecture():
    f=read(T/'ssi_siamese_architecture_selection__all_candidate_summary.csv');fig,axs=plt.subplots(2,4,figsize=(7.16,4.0),sharex=True)
    for r,cond in enumerate(['filtered','author_compatible']):
        for c,h in enumerate(H):
            g=f.loc[(f.outer_holdout==h)&(f.data_condition==cond)].sort_values('candidate');ax=axs[r,c]
            ax.plot(range(4),g.mean_validation_mae,'o-',lw=.7,ms=4,color='#0072B2')
            for x,row in enumerate(g.itertuples()):
                if row.selected:ax.plot(x,row.mean_validation_mae,'o',ms=7,mfc='none',mec='#D55E00',mew=1.1)
            ax.set_xticks(range(4),['A','B','C','D']);ax.set_title(N[h] if r==0 else '')
            if c==0:ax.set_ylabel(('Filtered' if r==0 else 'Author-compatible')+'\nMean inner MAE')
            if r==1:ax.set_xlabel('Candidate')
    fig.tight_layout();save(fig,'supp03_architectures')

def seeds():
    f=read(T/'siamese_seed_metrics.csv');fig,axs=plt.subplots(2,4,figsize=(7.16,4.0),sharex=True)
    for r,cond in enumerate(['filtered','author_compatible']):
        for c,h in enumerate(H):
            ax=axs[r,c]
            for role,col in [('familiar_test','#0072B2'),('unseen_test','#D55E00')]:
                g=f.loc[(f.outer_holdout==h)&(f.data_condition==cond)&(f.pair_role==role)&(f.metric=='mae')&f.seed.notna()].copy()
                if role=='familiar_test':
                    g=g.loc[g.dataset=='familiar_macro'] if (g.dataset=='familiar_macro').any() else g.groupby('seed',as_index=False).estimate.mean()
                ax.plot(g.seed,g.estimate,'o-' if role=='familiar_test' else 's--',lw=.7,ms=3,color=col,label='Familiar macro' if role=='familiar_test' else 'Withheld')
            ax.set_xticks([42,44,46]);ax.set_title(N[h] if r==0 else '')
            if c==0:ax.set_ylabel(('Filtered' if r==0 else 'Author-compatible')+'\nMAE')
            if r==1:ax.set_xlabel('Prespecified seed')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.05),ncol=2,frameon=False);fig.tight_layout();save(fig,'supp04_seed_variability')

def gdm_and_feature_shift():
    stats=read(T/'ssi_representation_statistics.csv');names=dict(zip(stats.feature_name,stats.author_abbreviation));params=read(T/'final_fit_and_seal__gdm_parameters.csv');curves=read(T/'gdm_transformation_curves.csv')
    f=params.drop_duplicates(['outer_holdout','data_condition','feature_name']);features=[f'ssi_{i:03d}' for i in range(60)]
    fig,axs=plt.subplots(1,2,figsize=(7.16,9.0),sharey=True)
    for k,cond in enumerate(['filtered','author_compatible']):
        matrix=f.loc[f.data_condition==cond].pivot(index='feature_name',columns='outer_holdout',values='total_coefficient_contribution').reindex(index=features,columns=H)
        im=axs[k].imshow(matrix,aspect='auto',cmap='viridis',vmin=0,vmax=f.total_coefficient_contribution.max());axs[k].set_xticks(range(4),['India','Malaysia','Taiwan','USA']);axs[k].set_yticks(range(60),[f'{x[-3:]}  {names[x]}' for x in features],fontsize=6);axs[k].set_title('Filtered' if k==0 else 'Author-compatible')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Sum of fitted I-spline coefficients',shrink=.55,pad=.03);save(fig,'supp05_gdm_contributions')
    for cond in ['filtered','author_compatible']:
        for page in range(4):
            fig,axs=plt.subplots(5,3,figsize=(7.16,8.5))
            for i,feature in enumerate(features[page*15:(page+1)*15]):
                ax=axs.flat[i]
                for h,col,style in zip(H,['#0072B2','#009E73','#D55E00','#CC79A7'],['-','--',':','-.']):
                    g=curves.loc[(curves.data_condition==cond)&(curves.outer_holdout==h)&(curves.feature==feature)]
                    ax.plot(g.raw_ssi_value,g.transformed_value,lw=.9,color=col,ls=style,label=N[h])
                ax.set_title(f'{feature[-3:]} {names[feature]}',fontsize=8);ax.ticklabel_format(style='sci',axis='x',scilimits=(-2,3));ax.tick_params(labelsize=6)
                if i%3==0:ax.set_ylabel('Spline transform',fontsize=7)
                if i>=12:ax.set_xlabel('Raw SSI value (native scale)',fontsize=7)
            fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.015),ncol=4,frameon=False,fontsize=7);fig.tight_layout(h_pad=1.4);save(fig,f'supp06_gdm_{cond}_{page+1}')
    f=read(T/'failure_decomposition.csv');f=f.loc[(f.metric=='ssi_ks_statistic')&(f.pair_role=='unseen_test')]
    fig,axs=plt.subplots(1,2,figsize=(7.16,9),sharey=True)
    for k,cond in enumerate(['filtered','author_compatible']):
        matrix=f.loc[f.data_condition==cond].pivot(index='feature',columns='outer_holdout',values='estimate').reindex(index=features,columns=H)
        im=axs[k].imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='cividis');axs[k].set_xticks(range(4),['India','Malaysia','Taiwan','USA']);axs[k].set_yticks(range(60),[f'{x[-3:]} {names[x]}' for x in features],fontsize=6);axs[k].set_title('Filtered' if k==0 else 'Author-compatible')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Source vs withheld-event KS statistic',shrink=.55,pad=.03);save(fig,'supp07_feature_shift')

def temporal():
    f=read(T/'target_stability.csv');bins=['same_date','1_to_7_days','8_to_30_days','31_to_90_days','more_than_90_days'];fig,axs=plt.subplots(1,3,figsize=(7.16,2.9),sharey=True)
    for k,h in enumerate(H[:3]):
        ax=axs[k]
        for c,cond in enumerate(['filtered','author_compatible']):
            g=f.loc[(f.dataset==h)&(f.data_condition==cond)&(f.metric=='mean')].set_index('date_gap_group')
            for i,b in enumerate(bins):
                row=g.loc[b]
                if np.isfinite(row.estimate):
                    ax.plot(i+(-.08 if c==0 else .08),row.estimate,'o' if c==0 else 's',color='#0072B2' if c==0 else '#999999',ms=4,mfc=('#0072B2' if c==0 else '#999999') if row.sufficiently_populated else 'none')
                    ax.text(i+(-.11 if c==0 else .11),row.estimate+(.025 if c==0 else -.055),str(int(row.n_pairs)),ha='center',fontsize=5.5,color='#0072B2' if c==0 else '#777777')
        ax.set_xticks(range(5),['Exact\ntime','1-7 d','>7-30 d','>30-90 d','>90 d'],rotation=35,ha='right');ax.set_title(N[h]);ax.set_ylim(0,1.02)
    axs[0].set_ylabel('Mean observed Jaccard distance')
    fig.legend([Line2D([],[],marker='o',color='#0072B2',ls=''),Line2D([],[],marker='s',color='#999999',ls='')],['Filtered','Author-compatible'],loc='lower center',bbox_to_anchor=(.5,-.07),ncol=2,frameon=False)
    fig.tight_layout();save(fig,'supp08_temporal_stability')
