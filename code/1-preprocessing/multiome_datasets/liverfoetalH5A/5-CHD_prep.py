# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

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

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / "transcripts.pkl").open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc / "atac_fragments_H7A.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(
    transcripts, fragments_file
)

# %%
np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max()).shape

# %%
plt.scatter(
    adata.var["means"],
    np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max())[
        adata.var.index
    ],
)
np.corrcoef(
    adata.var["means"],
    np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max() + 1)[
        adata.var.index
    ],
)

# %%
pickle.dump(
    selected_transcripts, (folder_data_preproc / "selected_transcripts.pkl").open("wb")
)

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
adata.obs["cell_original"] = adata.obs.index
adata.obs.index = adata.obs.index.str.split("-").str[0] + "-1"
adata.obs['batches'] = adata.obs['batch'].cat.codes.astype(int)

# %% [markdown]
# ### Create transcriptome for top 5000 genes

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(
    adata[:, adata.var.sort_values("dispersions_norm").tail(5000).index],
    path=dataset_folder / "transcriptome",
    overwrite=True
)

# %%
sc.pl.umap(adata, color=adata.var.index[(adata.var["symbol"] == "GLUL")][0])
sc.pl.umap(adata, color=["leiden", "celltype"])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Clustering

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome")

# %%
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings" / "cluster",
    overwrite=True
)

# %% [markdown]
# ## Folds

# %%
folds = chd.data.folds.Folds(folder_dataset / "folds")
folds.folds = [
    {
        "cells_train": np.arange(len(transcriptome.obs)),
        "cells_test": np.arange(len(transcriptome.obs))[:500],
        "cells_validation": np.arange(len(transcriptome.obs))[:500],
    }
]

# %% [markdown]
# ### Create transcriptome for all genes

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(
    adata,
    path=dataset_folder / "transcriptome_all",
    overwrite=True
)

# %%
sc.pl.umap(adata, color=adata.var.index[(adata.var["symbol"] == "GLUL")][0])
sc.pl.umap(adata, color=["leiden", "celltype"])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions_all" / "10k10k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments_all" / "10k10k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Clustering

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome_all")

# %%
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings_all" / "cluster",
    overwrite=True
)

# %% [markdown]
# ## Folds

# %%
folds = chd.data.folds.Folds(folder_dataset / "folds_all")
folds.folds = [
    {
        "cells_train": np.arange(len(transcriptome.obs)),
        "cells_test": np.arange(len(transcriptome.obs))[:500],
        "cells_validation": np.arange(len(transcriptome.obs))[:500],
    }
]
# %%
