
# %%
import numpy as np
import pandas as pd
import pickle
import pathlib
import random
import tqdm.auto as tqdm
import io
import scipy.sparse

import scanpy as sc
import chromatinhd as chd


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import legend
import seaborn as sns

import polyptich as pp
pp.setup_ipython()
import eyck

# %%
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.n_jobs = 10

# %%
dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"
folder_dataset = chd.get_output() / "datasets" / dataset_name
folder_plots = chd.get_output() / "datasets" / dataset_name / "plotsTOP100"
folder_plots.mkdir(exist_ok=True, parents=True)

# %%
##################################################
##### LOAD UMAP
##################################################
import pickle
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data_preproc = folder_data / dataset_name
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
adata = adata[adata.obs["celltype"].isin(["HSCs", "MEP", "Early_Ery", "Mid_Ery", "Late_Ery", "Early_MK", "Late_MK"])]
print(adata.shape)
# (3701, 27324)

sc.pl.umap(
    adata, 
    color=['leiden','celltype'],
    legend_loc="on data",
    legend_fontsize = 6,
    show=False
)

# # %% [markdown]
# # ### HVG

# # %%
# adata.X = adata.layers["log"]
# sc.pp.highly_variable_genes(adata, min_mean=0.00125, max_mean=3, min_disp=0.5, n_top_genes= 2000, batch_key="sample")
# sc.pl.highly_variable_genes(adata)
# print('Number of HVGs: {:d}'.format(adata[:, adata.var["highly_variable"]].n_vars))

# # %% [markdown]
# # ### PCA

# # %%
# adata.X = adata.layers["scale"]
# sc.pp.pca(adata, n_comps= 50, svd_solver='auto') # By default uses only highly variable genes if available
# sc.pl.pca(adata)
# sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)

# sc.pl.pca(
#     adata,
#     color=["sample", "sample", "batch", "batch", "pct_counts_mt", "pct_counts_mt"],
#     dimensions=[(0, 1), (2, 3), (0, 1), (2, 3), (0, 1), (2, 3)],
#     ncols=2,
# )

# # %% [markdown]
# # ### Harmony

# import scanpy.external as sce
# sce.pp.harmony_integrate(adata, key='sample', theta=1)

# adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']

# sc.pl.pca(
#     adata,
#     color=["sample", "sample", "batch", "batch", "pct_counts_mt", "pct_counts_mt"],
#     dimensions=[(0, 1), (2, 3), (0, 1), (2, 3), (0, 1), (2, 3)],
#     ncols=2,
# )

# # %% [markdown]
# # ### UMAP

# # %%
# sc.pp.neighbors(adata, 
#                 n_pcs=11, 
#                 use_rep="X_pca", 
#                 knn=True, 
#                 random_state=42, 
#                 method="umap", 
#                 metric="euclidean")
# sc.tl.umap(adata)
# sc.pl.umap(
#     adata,
#     color="sample",
#     size=2,
# )

# # %%
# import magic
# magic_operator = magic.MAGIC(knn=30, solver="approximate")
# X_smoothened = magic_operator.fit_transform(adata.X)
# X_smoothened[X_smoothened < 0] = 0
# adata.layers["magic"] = X_smoothened


# # %% [markdown]
# # ## Interpret and subset

# # %%
# sc.tl.leiden(adata,resolution=0.2, flavor="igraph", n_iterations=-1)
# sc.pl.umap(
#     adata, 
#     color=['leiden'],
#     legend_loc="on data",
#     legend_fontsize = 6,
#     show=False
# )

# # %% [markdown]
# # ## Re-assesing quality control and cell filtering

# # %%
# sc.pl.umap(
#     adata,
#     color=["leiden", "doublet_score", "sample", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
#     # increase horizontal space between panels
#     wspace=0.5,
#     ncols=2,
# )

# # %% [markdown]
# # ## Annotation

# # %%
# sc.tl.rank_genes_groups(
#     adata, "leiden", method="wilcoxon", key_added="wilcoxon", use_raw=False
# )

# # %%
# # Extract the dictionary of gene names from recarray
# top_genes = {cluster: adata.uns["wilcoxon"]["names"][cluster] for cluster in adata.uns["wilcoxon"]["names"].dtype.names}

# # Convert to DataFrame
# num_top_genes = 20  # Adjust the number of genes to retrieve
# top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# # %%
# print(top_genes_df)

# %%
###################################
### DE genes
###################################
columnOI='celltype'
markersMACRO = eyck.m.t.diffexp.compare_two_groups(adata, adata.obs['celltype'].isin(["Late_Ery", "Early_Ery"]), adata.obs['celltype'].isin(["HSCs", "MEP", "Early_MK", "Late_MK"]), use_raw=False, layer="magic")
markersMACRO = markersMACRO.loc[markersMACRO['pvals_adj']<=0.05]

# %%
diffexp = markersMACRO
# diffexp=diffexp.loc[diffexp['lfc']>0]

# %%
diffexp['logFC']=diffexp['lfc']
diffexp['gene']=diffexp['symbol']
diffexp['FDR']=diffexp['pvals_adj']

# %%
diffexp.shape
# (7050, 27324)

# %%
### Plot some genes
eyck.m.t.plot_umap(adata,["leiden", "batch"] + diffexp['symbol'].head(20).tolist(),datashader=False, cmap='coolwarm', ncol=4, layer="magic").display()

# %%
