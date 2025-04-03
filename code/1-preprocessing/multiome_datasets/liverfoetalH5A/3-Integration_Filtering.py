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
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=150, facecolor="white", figsize=[4,4])
ugent_palette = [
    "#1e62c7ff", "#3373ccff", "#4a83d1ff", "#78a1deff", "#6191d7ff",
    "#8eb1e3ff", "#a4c1e8ff", "#bbcfeeff", "#d2def3ff", "#e7eff9ff"
]
accent_colour = ["#2D8CA8"]
sns.set_palette(ugent_palette)

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
# ## Load in data

# %%
adata = pickle.load((folder_data_preproc / "adata_QC.pkl").open("rb"))
transcripts = pickle.load((folder_data_preproc / "transcripts.pkl").open("rb"))

# %%
fragments_data = pickle.load((folder_data_preproc / "atac_pandas.pkl").open("rb"))

# %%
fragment_counts = fragments_data.groupby('sequence_id')['count'].sum().reset_index()

# %% [markdown]
# ## Filtering Cascade
# %%
adata.obs["filter"] = "kept"
adata_filtered = adata.copy()
original_cells = adata.obs.index

# Filter based on RNA unique molecular identifiers (UMIs)
sc.pp.filter_cells(adata_filtered, min_counts=100)

removed_cells_min_counts = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_min_counts), 'filter'] = 'removed_min_counts'
original_cells = adata_filtered.obs.index

print(f'Number of cells after min count filter: {adata_filtered.n_obs}')
print(f'Number of genes after min count filter: {adata_filtered.n_vars}')

# %%
# Filter based on expressed genes
sc.pp.filter_cells(adata_filtered, min_genes=250)

removed_cells_min_genes = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_min_genes), 'filter'] = 'removed_min_genes'
original_cells = adata_filtered.obs.index 

print(f'Number of cells after min gene filter: {adata_filtered.n_obs}')
print(f'Number of genes after min gene filter: {adata_filtered.n_vars}')

# %%
# Filter based on mitochondrial read fraction
adata_filtered = adata_filtered[adata_filtered.obs['pct_counts_mt'] < 40]

removed_cells_mt_fraction = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_mt_fraction), 'filter'] = 'removed_mt_fraction'
original_cells = adata_filtered.obs.index

print(f'Number of cells after MT fraction filter: {adata_filtered.n_obs}')
print(f'Number of genes after MT fraction filter: {adata_filtered.n_vars}')

# %%
# Filter based on number of cells expressed for each gene
sc.pp.filter_genes(adata_filtered, min_cells=3)

removed_cells_min_cells_genes = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_min_cells_genes), 'filter'] = 'removed_min_cells_genes'
original_cells = adata_filtered.obs.index

print(f'Number of cells after cell filter: {adata_filtered.n_obs}')
print(f'Number of genes after cell filter: {adata_filtered.n_vars}')

# %%
# Filter based on Doublet detection
adata_filtered = adata_filtered[~adata_filtered.obs["predicted_doublet"].astype(bool)]

removed_cells_min_cells_genes = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_min_cells_genes), 'filter'] = 'removed_doublets'
original_cells = adata_filtered.obs.index

print('Number of cells after doublet removal: {:d}'.format(adata_filtered.n_obs))
print('Number of genes after doublet removal: {:d}'.format(adata_filtered.n_vars))

# %%
# # Filter based on at least 1000 fragments per cell
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
filtered_fragment_counts = fragment_counts[fragment_counts['count'] > 1000]
filtered_sequence_ids = filtered_fragment_counts['sequence_id']
adata_filtered = adata_filtered[adata_filtered.obs_names.isin(filtered_sequence_ids)]

removed_cells_min_cells_genes = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_min_cells_genes), 'filter'] = 'removed_low_fragments'
original_cells = adata_filtered.obs.index

print('Number of cells with at least 1000 fragments per cell: {:d}'.format(adata_filtered.n_obs))
print('Number of genes with at least 1000 fragments per cell: {:d}'.format(adata_filtered.n_vars))

# %%
# Filter based only genes with transcripts available
adata_filtered = adata_filtered[:, adata_filtered.var.index.isin(transcripts["ensembl_gene_id"])]

removed_cells_min_cells_genes = set(original_cells) - set(adata_filtered.obs.index)
adata.obs.loc[adata.obs.index.isin(removed_cells_min_cells_genes), 'filter'] = 'removed_no_transcripts'
original_cells = adata_filtered.obs.index

print('Number of genes after transcript filtering: {:d}'.format(adata_filtered.n_obs))
print('Number of genes after transcript filtering: {:d}'.format(adata_filtered.n_vars))

# # Filter based on TSS enrichment score of 3
# adata = adata[~adata.obs["TSS"].astype(bool)]
# print('Number of cells after TSS: {:d}'.format(adata.n_obs))
# print('Number of genes after TSS: {:d}'.format(adata.n_vars))

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
num_samples = len(samples)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), sharex=True, sharey=True)

color_map = {
    "kept": "#1E64C8",
    "removed_min_counts": "#FFD200",
    "removed_min_genes": "#8BBEE8",
    "removed_mt_fraction": "#27ABAD",
    "removed_min_cells_genes": "#2D8CA8",
    "removed_doublets": "#71A860",
    "removed_no_transcripts": "#AEB050",
    "removed_low_fragments": "#F1A42B"
}
axes = axes.flatten()
for ax, sample in zip(axes, samples):
    ATAC_counts = fragments_data.query("batch == @sample").groupby("sequence_id")["count"].sum()
    adata_subset = adata[adata.obs["batch"] == sample]
    mRNA_UMIs = adata_subset.X.sum(axis=1).A1
    mRNA_cells = pd.Series(mRNA_UMIs, index=adata_subset.obs_names)
    common_cells = mRNA_cells.index.intersection(ATAC_counts.index)
    mRNA_UMIs_matched = mRNA_cells.loc[common_cells]
    ATAC_fragments_matched = ATAC_counts.loc[common_cells]

    categories = adata_subset.obs.loc[common_cells, "filter"]
    colors = categories.map(color_map)

    ax.scatter(mRNA_UMIs_matched, ATAC_fragments_matched, c=colors, alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f"Sample {sample}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

fig.supxlabel('Gene Counts (log scale)')
fig.supylabel('ATAC fragments (log scale)')
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=6, label=label)
                  for label, color in color_map.items()]
fig.legend(handles=legend_handles, title="Filtering Steps", loc="upper left", fontsize=8, frameon=True)

plt.tight_layout()
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.85, top=0.9, hspace=0.4, wspace=0.4)
plt.show()

# %%
pickle.dump(adata_filtered, (folder_data_preproc / "adata_filtered.pkl").open("wb"))

# %%
