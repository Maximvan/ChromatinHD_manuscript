
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
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
adata = adata[adata.obs["celltype"].isin(["Monocyte_I", "Monocyte_II", "Macrophage_I", "Macrophage_II", "Macrophage_III"])]

# %%
sc.pl.umap(adata, color=["celltype"])

# %%
grouped_palette = [
    '#cc4c02',  # Monocyte
    '#fe9929',  # pre-Macrophage
    '#fdd49e',  # Macrophage
]
desired_order = [
    'Monocyte', 
    'Pre-Macrophage',
    'Macrophage'
]

# %%
# saving KC monocyte subset
adata_sn_kc = adata[adata.obs["celltype"].isin(["Monocyte_I", "Monocyte_II", "Macrophage_I", "Macrophage_II", "Macrophage_III"])]
cluster_to_celltype = {
    'Monocyte_I': 'Monocyte',
    'Monocyte_II': 'Monocyte',
    'Macrophage_I': 'Pre-Macrophage',
    'Macrophage_II': 'Macrophage',
    'Macrophage_III': 'Macrophage',
}
adata_sn_kc.obs['annotation'] = adata.obs['celltype'].map(cluster_to_celltype)
adata_sn_kc.obs['annotation'] = adata_sn_kc.obs['annotation'].astype('category')
adata_sn_kc.obs['annotation'] = adata_sn_kc.obs['annotation'].cat.reorder_categories(desired_order)
sc.pl.umap(adata_sn_kc, color=["annotation"], palette=grouped_palette,  frameon=False, legend_loc='left margin')
sc.pl.umap(adata_sn_kc, color=["celltype"])

# %%
pickle.dump(adata_sn_kc, (folder_data_preproc / "adata_sn_kc_annotated.pkl").open("wb"))

# %%
