# %%
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import scanpy as sc

import polyptich as pp
from script_functions import *

pp.setup_ipython()

# %%
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=150, facecolor="white", figsize=[4,4])

# %%
###################################
### General
###################################
import os
os.getcwd()

baseFolder='/srv/data/liesbetm/Projects/u_mgu/Liesbet/seurat_vs_scanpy/'
saveFolder=baseFolder+'results/scanpy/'


# %%
##################################################
##### LOAD DATA
##################################################
listLabels=['ABU2','ABU7','ABU9','CS141','CS142','CS144']

adatas=[]
for nameSample in listLabels:
    print('Loading '+nameSample)
    adataTmp = sc.read_10x_mtx(baseFolder+"rawData/"+nameSample,
                        var_names="gene_symbols",
                        cache=True)
    adataTmp.var_names_make_unique()
    adataTmp.obs['sample']=nameSample
    adataTmp.obs.index=adataTmp.obs.index+'-'+nameSample
    adatas.append(adataTmp)
adatas

# %%
### Merge data
import anndata as ad
adata = ad.concat(adatas, join="inner", label="batch")
adata

# %%
### Plot highest expressed genes
# sc.pl.highest_expr_genes(adata, n_top=20)


# %%
##################################################
##### QC
##################################################
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

### Calculate percent.mito
adata.var["mt"] = adata.var_names.str.startswith("mt-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

# %%
###################################
### Mad
###################################

### Function MAD
def identify_mad_outliers(series, cutOffLow=3, cutOffHigh=3):
    median = series.median()
    mad = np.median(np.abs(series - median))
    upper_limit = median + cutOffHigh * mad
    lower_limit = median - cutOffLow * mad
    return series[(series > upper_limit) | (series < lower_limit)]

### Get outliers
outliers_n_genes = identify_mad_outliers(adata.obs['n_genes'],2,4)
outliers_total_counts = identify_mad_outliers(adata.obs['total_counts'],3,3)
outliers_pct_counts_mt = identify_mad_outliers(adata.obs['pct_counts_mt'],3,3)

adata.obs['nGene.drop']=False
adata.obs.loc[adata.obs.index.isin(outliers_n_genes.index),'nGene.drop']=True
adata.obs['nUMI.drop']=False
adata.obs.loc[adata.obs.index.isin(outliers_total_counts.index),'nUMI.drop']=True
adata.obs['mito.drop']=False
adata.obs.loc[adata.obs.index.isin(outliers_pct_counts_mt.index),'mito.drop']=True

adata.obs.loc[adata.obs['n_genes']<1000,'nGene.drop']=True

### Finalize
adata.obs['final.drop'] = adata.obs['nGene.drop'] | adata.obs['nUMI.drop'] | adata.obs['mito.drop']
adata.obs['final.drop'].value_counts()


# %%
##################################################
##### PLOTS
##################################################
# ### Create barplot
# fig=createBarplot(adata)
# fig.savefig(saveFolder+'1_barplot.png', dpi=150)

# # %%
# ### vlnPlot before filtering
# fig=createVlnPlot(adata)
# fig.savefig(saveFolder+'2a_vlnPlot_beforeFiltering.png', dpi=150)

# # %%
# ### vlnPlot after filtering
# adataSlice=adata[adata.obs['final.drop']==False].copy()
# fig=createVlnPlot(adataSlice)
# fig.savefig(saveFolder+'2b_vlnPlot_afterFiltering.png', dpi=150)



# %%
##################################################
##### FINALIZE QC
##################################################
# adata=adata[adata.obs['final.drop']==False]

### Load metaData
adata.obs["cell"]=adata.obs.index
metaData=pd.read_csv(baseFolder+'rawData/metaData.csv',sep=',')

### Filter
print(set(metaData['cell']).difference(set(adata.obs['cell'])))
adata=adata[metaData['cell']]

### Add metadata info
tmp=pd.merge(adata.obs, metaData[['cell','type','annot']], on='cell', how="inner")
tmp=tmp.set_index('cell',drop=False)
tmp.index.name = None
adata.obs=tmp


# %%
##################################################
##### NORMALIZE
##################################################
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %%
##################################################
##### HVG
##################################################
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# sc.pl.highly_variable_genes(adata)

# %%
adataFull=adata.copy()
adata = adata[:, adata.var.highly_variable]

# %%
##################################################
##### SCALE
##################################################

# ### Regress out
# sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])

### Scale
# sc.pp.scale(adata, max_value=10)

# %%
##################################################
##### PCA
##################################################
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.pca(adata) # ??? dit is hoe Wouter het doet - wat is verschil tussen sc.pp en sc.tl??
sc.pl.pca(adata, color="sample")

# %%
### Choose PCs
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)


# %%
##################################################
##### UMAP
##################################################
sc.pp.neighbors(adata, n_pcs=40)
sc.tl.umap(adata)

# %%
### Clustering
sc.tl.leiden(adata,resolution=1.0)

# %%
### Plot umap
sc.pl.umap(
    adata, 
    color=['leiden'],
    legend_loc="on data",
    legend_fontsize = 6,
    show=False
)

# %%
##################################################
##### PLOTS
##################################################

############### FeaturePlot ###############
fig=FeaturePlot(adata, 'Clec4f','coolwarm')
# fig.savefig(saveFolder+'Clec4f.png',dpi=300)

# %%
fig=FeaturePlotPP(adata, 'Clec4f','coolwarm')
fig.display()
# fig.savefig(saveFolder+'Clec4f_v2.png',dpi=300)

# %%
############### Plot umap ###############
figsize=5
wspace = 0.5
nrows=2
ncols=2
calcFigSize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize)

fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=calcFigSize)

sc.pl.umap(
    adata, 
    color=['leiden'],
    legend_loc="on data",
    legend_fontsize = 6,
    ax=axes[0,0],
    show=False
)
sc.pl.umap(
    adata, 
    color=['annot'],
    legend_loc="on data",
    legend_fontsize = 6,
    ax=axes[0,1],
    show=False
)
sc.pl.umap(
    adata, 
    color=['sample'],
    legend_loc="right margin",
    legend_fontsize = 6,
    ax=axes[1,0],
    show=False
)
sc.pl.umap(
    adata, 
    color=['type'],
    legend_loc="right margin",
    legend_fontsize = 6,
    ax=axes[1,1],
    show=False
)
fig.tight_layout()
fig.savefig(saveFolder+'3_umap.png',dpi=300)

# %%
############### Plot mito ###############
figsize=5
wspace = 0.5
nrows=1
ncols=3
calcFigSize=(ncols * figsize + figsize * wspace * (ncols - 1), nrows * figsize)

fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=calcFigSize)

sc.pl.umap(
    adata, 
    color=['n_genes'],
    legend_fontsize = 6,
    ax=axes[0],
    show=False
)
sc.pl.umap(
    adata, 
    color=['total_counts'],
    legend_fontsize = 6,
    ax=axes[1],
    show=False
)
sc.pl.umap(
    adata, 
    color=['pct_counts_mt'],
    legend_fontsize = 6,
    ax=axes[2],
    show=False
)
fig.tight_layout()
fig.savefig(saveFolder+'4_plot_mito.png',dpi=300)

# %%
# ############### Violin plot gene ###############
# sc.settings.set_figure_params(dpi=150, facecolor="white", figsize=[8,8])
# sc.pl.violin(adata, ["Cd5l"], groupby="annot", rotation=90)
# sc.pl.violin(adata, ["Cd5l"], groupby="leiden", rotation=90)

# sc.settings.set_figure_params(dpi=150, facecolor="white", figsize=[4,4])

# %%
############### Split plot advanced ###############
tmp=adata.obs['type'].unique().tolist()
tmp.sort()
colDict = dict(zip(tmp, pythonColors[:len(tmp)]))
fig=splitPlotAdvanced(adata,'type',panelSize=10,ncols=2,colDict=colDict)
fig.savefig(saveFolder+'5a_split_type.png',dpi=300)

# %%
tmp=adata.obs['sample'].unique().tolist()
tmp.sort()
colDict = dict(zip(tmp, pythonColors[:len(tmp)]))
fig=splitPlotAdvanced(adata,'sample',panelSize=10,ncols=3,colDict=colDict)
fig.savefig(saveFolder+'5b_split_sample.png',dpi=300)

# %%
###################################
### Annotated umap
###################################
sc.pl.umap(adata,color='annot',legend_loc='on data',legend_fontsize=5)

# %%
colorsAnnot = {
    'Endothelial cells':'#fbb062',
    'Fibroblasts':'#a41d2b',
    'Hepatocytes':'#ea579f',
    'Kupffer cells':'#5da6db',
    'Cholangiocytes':'#c61b84',
    'Monocytes & Monocyte-derived cells':'#a4daf3',
    'Neutrophils':'#b4b5b5',
    'T cells':'#3ab04a',
    'NK cells':'#4a6e34',
    'ILC1s':'#a3d7ba',
    'Meso':'#d0110b',
    'cDC2s':'#bf00ff',
    'cDC1s':'#893a86',
    'Mig. cDCs':'#702963',
    'pDCs':'#9c7eba',
    'B cells':'#914d22',
    'Plasma cells':'#a7704e',
    'Peritoneal Macs':'#516fb9',
    'Basophils':'#191919',
    'HsPCs':'#f19fc3'
}
set(colorsAnnot.keys()).difference(set(adata.obs['annot'].unique()))
set(adata.obs['annot'].unique()).difference(set(colorsAnnot.keys()))

### Prepare plot
toPlot=pd.DataFrame(adata.obsm['X_umap'])
toPlot.columns=['UMAP_1','UMAP_2']
toPlot.index=adata.obs.index
toPlot['annot'] = adata.obs['annot']
toPlot['color'] = adata.obs['annot'].map(colorsAnnot)

cluster_centers = toPlot.groupby('annot')[['UMAP_1', 'UMAP_2']].median()

### Make plot
fig = pp.Figure()
ax = fig.main.add(pp.Panel((3,3)))
ax.scatter(
    toPlot["UMAP_1"], 
    toPlot["UMAP_2"], 
    c = toPlot["color"],  
    s=1,
    marker = ".",
    edgecolor = "none"
)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.axis('off')

for annot, (x_center, y_center) in cluster_centers.iterrows():
    ax.text(x_center, y_center, annot, fontsize=5, fontweight='bold', 
            ha='center', va='center', color='black')

# fig.display()
fig.savefig(saveFolder+'umapAnnot.png',dpi=300)



# %%
##################################################
##### DE GENES - all clusters vs all clusters
##################################################
columnOI='annot'

##### Calculate DE genes #####
sc.tl.rank_genes_groups(adata, columnOI, groups='all', 
                        reference='rest', method='wilcoxon', 
                        n_genes=adata.shape[1], pts=True)

# %%
##### Get DE genes #####
cellTypes=adata.obs[columnOI].unique()
DEgenesAll = pd.DataFrame()
for cellType in cellTypes:
    DEgenesTmp = sc.get.rank_genes_groups_df(adata,group=cellType)
    ##Filter and sort
    DEgenesTmp = DEgenesTmp[(DEgenesTmp.logfoldchanges > 1) & (DEgenesTmp.pvals_adj < 0.01)]
    DEgenesTmp=DEgenesTmp.sort_values(by='scores',ascending=False)
    DEgenesTmp['cellType']=cellType
    ##Get pct    
    cellsGroup1=adata.obs.loc[adata.obs[columnOI]==cellType,'cell']
    cellsGroup2=adata.obs.loc[adata.obs[columnOI]!=cellType,'cell']
    genesOI=DEgenesTmp['names']
    DEgenesTmp['pct.1']=np.array(np.mean(adataFull[cellsGroup1, genesOI].X.todense()>0, axis=0))[0]
    DEgenesTmp['pct.2']=np.array(np.mean(adataFull[cellsGroup2, genesOI].X.todense()>0, axis=0))[0]
    ##Merge
    DEgenesAll=pd.concat([DEgenesAll, DEgenesTmp],ignore_index=True)

# %%
##### Write to Excel #####
import xlsxwriter

fileName = saveFolder+'DEgenes_allClusters.xlsx'
writer = pd.ExcelWriter(fileName, engine='xlsxwriter')

sheets=adata.obs[columnOI].unique()
for sheet in sheets:
    tmp=DEgenesAll[DEgenesAll.cellType==sheet]
    tmp.index=range(1,tmp.shape[0]+1)
    if(len(sheet)>30):
        sheet=sheet[0:30]
    tmp.to_excel(writer, sheet_name=sheet)
writer.close()


# %%
##################################################
##### DE GENES - some clusters vs some clusters
##################################################

##### Calculate DE genes #####
def getDEgenes(adataTmp,obsColumnName,group1,group2):
    sc.tl.rank_genes_groups(adataTmp, obsColumnName, groups=[group1], reference=group2, method='wilcoxon', n_genes=adataTmp.shape[1], pts=True)
    DEgenesTmp = sc.get.rank_genes_groups_df(adataTmp, group=group1)
    ##Filter and sort
    DEgenesTmp = DEgenesTmp[(DEgenesTmp.logfoldchanges > 1) | (DEgenesTmp.logfoldchanges < -1) & (DEgenesTmp.pvals_adj < 0.01)]
    DEgenesTmp=DEgenesTmp.sort_values(by='scores',ascending=False)
    ##Get pct    
    cellsGroup1=adataTmp.obs.loc[adataTmp.obs[obsColumnName]==group1,'cell']
    cellsGroup2=adataTmp.obs.loc[adataTmp.obs[obsColumnName]==group2,'cell']
    genesOI=DEgenesTmp['names']
    DEgenesTmp['pct.1']=np.array(np.mean(adataFull[cellsGroup1, genesOI].X.todense()>0, axis=0))[0]
    DEgenesTmp['pct.2']=np.array(np.mean(adataFull[cellsGroup2, genesOI].X.todense()>0, axis=0))[0]
    return(DEgenesTmp)

##### Get DE genes #####
columnOI='annot'

KC_vs_EC=getDEgenes(adata,columnOI,'Kupffer cells','Endothelial cells')
KC_vs_Cholang=getDEgenes(adata,columnOI,'Kupffer cells','Cholangiocytes')

# %%
##### Merge DE genes #####
KC_vs_EC['comp']='KC_vs_EC'
KC_vs_Cholang['comp']='KC_vs_Cholang'
DEgenesAll=pd.concat([KC_vs_EC, KC_vs_Cholang],ignore_index=True)

# %%
##### Write to Excel #####
import xlsxwriter
fileName = saveFolder+'DEgenes_someClusters.xlsx'
writer = pd.ExcelWriter(fileName, engine='xlsxwriter')

sheets=DEgenesAll['comp'].unique()
for sheet in sheets:
    tmp=DEgenesAll[DEgenesAll.comp==sheet]
    tmp=tmp.drop('comp', axis=1)
    tmp.index=range(1,tmp.shape[0]+1)
    if(len(sheet)>30):
        sheet=sheet[0:30]
    tmp.to_excel(writer, sheet_name=sheet)
writer.close()

# %%
##################################################
##### HEATMAP
##################################################

def getCells(listCelltypes, obsColumn, maxAmountCells):
    cellsToReturn=[]
    for cellName in listCelltypes:
        cellsTmp=adata.obs.loc[adata.obs[obsColumn]==cellName].index.tolist()
        cellsTmp=getRandomCells(cellsTmp, maxAmountCells)
        print(len(cellsTmp))
        cellsToReturn += cellsTmp

    return cellsToReturn

########## Genes ##########
genesOI=KC_vs_EC['names'].head(10).tolist() + KC_vs_EC['names'].tail(10).tolist()

### Annot genes
var = pd.DataFrame(index = genesOI)
var["color"] = ["blue"] * 10 + ['red'] *10
var['module']=["group1"] * 10 + ["group2"] * 10

orderGenes=['group1','group2']

########## Cells ##########
maxNrCells=1000
orderCells=['Kupffer cells','Endothelial cells']
colorCells=['green','orange']
cellsOI=getCells(orderCells,'annot',maxNrCells)

colorsAnnot = pd.DataFrame({
    'cellType': orderCells,
    'color': colorCells
})
colorsAnnot=colorsAnnot.set_index('cellType')

### Annot cells
obs=pd.DataFrame(index = cellsOI)

colCellType=[]
for cellType in orderCells:
    cellsTmp=adata.obs.loc[adata.obs['annot']==cellType].index.tolist()
    if(len(cellsTmp)>maxNrCells):
        toAdd=[cellType] * maxNrCells
    else:
        toAdd=[cellType] * len(cellsTmp)
    colCellType += toAdd
obs['cellType']=colCellType



########## Data ##########
data=pd.DataFrame(adata[cellsOI, genesOI].X.todense())

### Scale data to 0-1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

data.columns=genesOI
data.index=cellsOI


# %%
########## Plot heatmap ##########
fig=createHeatmap(data, orderCells, orderGenes, var, obs, colorsAnnot)
fig.savefig(saveFolder+'heatmap.png',dpi=300)


# %%
###################################
### Save
###################################
### Save before taking HVG
import pickle
pickle.dump(adata, open(baseFolder+"results/Pkls/adata.pkl", "wb"))
pickle.dump(adataFull, open(baseFolder+"results/Pkls/adataFull.pkl", "wb"))


# %%
###################################
### Reload
###################################
### Load data again
import pickle
adata = pickle.load(open(baseFolder+"results/Pkls/adata.pkl", "rb"))
adataFull = pickle.load(open(baseFolder+"results/Pkls/adataFull.pkl", "rb"))

sc.pl.umap(
    adata, 
    color=['leiden'],
    legend_loc="on data",
    legend_fontsize = 6,
    show=False
)



# %%
