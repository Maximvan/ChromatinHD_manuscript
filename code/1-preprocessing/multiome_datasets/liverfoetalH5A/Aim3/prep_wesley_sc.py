
# %%
import os
import sys

import polyptich as pp
pp.setup_ipython()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pickle

import scanpy as sc

import polyptich as pp
import eyck

# %%
folder_data = pp.paths.get_output() / "data" / "Wesley22_KC"
folder_data.mkdir(exist_ok=True, parents=True)

# %%
h5ad_file = folder_data / "adata.h5ad"
# h5ad_file.unlink()
if not h5ad_file.exists():
    # # !wget https://datasets.cellxgene.cziscience.com/66e520ac-ee32-4f34-a55b-ab99ca6fda80.h5ad -O {h5ad_file}
    os.system(
        f"wget https://storage.googleapis.com/haniffalab/liver-development/primary_cell_types/Kupffer_primary_ad.h5ad -O {h5ad_file}"
    )

# %%
adata = sc.read_h5ad(h5ad_file)

# %%
adataF = adata[adata.obs["time"].isin([
                                        '11+3',
                                        #'12'
                                        ])]                              

# %%
sc.pl.umap(adataF, color=["time", "source"], frameon=False, show=True)

# %%
# PCA
sc.pp.pca(adataF, n_comps=50)
sc.pl.pca(adataF, color=["time", "source"], frameon=False, show=True)   

#%%
# HVG
sc.pp.highly_variable_genes(adataF, n_top_genes=2000)
sc.pl.highly_variable_genes(adataF, show=True)

#%%
# UMAP
sc.pp.neighbors(adataF, n_neighbors=10, n_pcs=15)
sc.tl.umap(adataF)
sc.pl.umap(adataF, color=["time", "source"], frameon=False, show=True)  


# %%
# gene plotting
gene = "CDH5"
sc.pl.umap(adataF, color=[gene, 'time'], frameon=False, show=True)
sc.pl.dotplot(adataF, gene, groupby="time", show=True)


# %%
