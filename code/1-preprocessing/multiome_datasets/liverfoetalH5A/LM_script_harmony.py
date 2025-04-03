# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

import scanpy as sc

import polyptich as pp
from script_functions import *

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

# %%
baseFolder='/srv/data/liesbetm/Projects/u_mgu/Liesbet/seurat_vs_scanpy/'
saveFolder=baseFolder+'results/harmonypy/'

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
### Finalize QC
###################################
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
##### PCA
##################################################
sc.tl.pca(adata, svd_solver="arpack")
sc.pl.pca(adata, color="sample")

# %%
### Choose PCs
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)


# %%
##################################################
##### HARMONY
##################################################
import scanpy.external as sce
sce.pp.harmony_integrate(adata, 'sample')

adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']


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
###################################
### Save
###################################
### Save before taking HVG
import pickle
pickle.dump(adata, open(baseFolder+"results/Pkls/adataHarmony.pkl", "wb"))


# %%
###################################
### Reload
###################################
### Load data again
import pickle
adata = pickle.load(open(baseFolder+"results/Pkls/adataHarmony.pkl", "rb"))

sc.pl.umap(
    adata, 
    color=['leiden'],
    legend_loc="on data",
    legend_fontsize = 6,
    show=False
)



# %%
