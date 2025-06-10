
# %%
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

import chromatinhd as chd

import magic
import eyck
import scipy.sparse

# %%
folder_data = chd.get_output() / "data"

# %%
def patched_setstate(self, state):
    self.__dict__ = state.copy()
    self.__dict__["_adata_ref"] = None  # Skip broken reference

import anndata._core.file_backing
anndata._core.file_backing.AnnDataFileManager.__setstate__ = patched_setstate

adata = pickle.load((folder_data / "mouse_ALK1KO" / "adata.pkl").open("rb"))


#%%
sc.pl.umap(adata, color=["leiden"])
adata = adata[~adata.obs["leiden"].isin(["9"])]
# saving KC monocyte subset
cluster_to_celltype = {
    '6': 'Monocyte',
    '10': 'Early Macrophage',
    '7': 'Early Macrophage',
    '0': 'Late Macrophage',
    '2': 'Late Macrophage',
    '3': 'Late Macrophage',
    '5': 'Late Macrophage',
    '1': 'Late Macrophage',
}
adata.obs['annotation'] = adata.obs['leiden'].map(cluster_to_celltype)
desired_order = [
    'Monocyte', 
    'Early Macrophage',
    'Late Macrophage'
]
adata.obs['annotation'] = adata.obs['annotation'].astype('category')
adata.obs['annotation'] = adata.obs['annotation'].cat.reorder_categories(desired_order)
sc.pl.umap(adata, color=["leiden", "annotation"], title=["leiden", "annotation"])


# %%
pickle.dump(adata, (folder_data / "mouse_ALK1KO" / "adata_mouse_kc_annotated.pkl").open("wb"))

# %%
