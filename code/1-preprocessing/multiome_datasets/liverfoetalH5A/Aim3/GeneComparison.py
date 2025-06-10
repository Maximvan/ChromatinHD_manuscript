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

# %%
adataSN = pickle.load((folder_data / dataset_name / "adata_sn_kc_annotated.pkl").open("rb"))

# %%
dataset_name = "mouse_ALK1KO"
adataM = pickle.load((folder_data / dataset_name / "adata_mouse_kc_annotated.pkl").open("rb"))

# %%
dataset_name = "suo22_kc"
adata8 = pickle.load((folder_data / dataset_name / "adata_sc_8_annotated.pkl").open("rb"))
adata11 = pickle.load((folder_data / dataset_name / "adata_sc_11_annotated.pkl").open("rb"))
adata14 = pickle.load((folder_data / dataset_name / "adata_sc_14_annotated.pkl").open("rb"))
adata17 = pickle.load((folder_data / dataset_name / "adata_sc_17_annotated.pkl").open("rb"))


# %%
gene_list =[
    "HLA-DQA1", "HLA-DQB1", "HLA-DRB1", "HLA-DPB1", "HLA-DRA", "HLA-DMB", "HLA-DPA1",  "HLA-DMA", "CD86", # antigen presentation and co-stimulation 
    "CTSB", "LRP1", "AXL", "RAB7A", "GAS6", "MERTK",  # phagocytosis and lysosomal activity
    "SIRPA", "HMOX1", "FTL", # erythroid remodeling/phagocytosis
     "IL1B", "NLRP3", "CCL3", "TNF", "CYBB",  "CXCL8", "P2RX7", # inflammasome
    "CASP1",  "NLRC4", "NFKB1", # pro-inflammatory cytokines
    'CLEC7A', 'AXL', 'CD163', 'MRC1', 'MSR1', 'MERTK', 'CD68', 'MARCO', 'CD36', 'CALR', 'TIMD4', 'TYRO3', 'APOL2', 'SCARA5', 'FCGR1A', # scavenger receptors
]

# %%
gene_list = [   
    "AHNAK", "JAML", "ITGAX", "ICAM1", "MARCKS", "VCAN", "ADGRE5", "ITGAM", "CD44", "ITGB2", "ITGAL", # from GO analysis
    "SIGLEC1", "ITGB5", "ADGRE1",  "SIGLEC11", "SDC3", "ITGA9", "ITGAD", "ADGRG6", "BCAM", "VCAM1", "CDH5", # DEGs from liver cell atlas involved in cell adhesion
    "GFRA2", "CD163", "HMOX1", "SLC1A3", "MARCO", "ADRB1", "TIMD4", "SUCNR1", "SLC40A1",  "SLC16A9", "FOLR2", "VCAM1", "TMEM26",  "CD5L" # KC marker genes
]

# %%
sc.pl.dotplot(
            adataSN, 
            var_names=gene_list,
            gene_symbols="symbol",
            groupby="annotation",
            layer="magic",
            swap_axes=True,
            )

# %%
sc.pl.dotplot(
            adata8, 
            var_names=gene_list,
            gene_symbols="feature_name",
            groupby="annotation",
            swap_axes=True,
            )
# %%
sc.pl.dotplot(
            adata11, 
            var_names=gene_list,
            gene_symbols="feature_name",
            groupby="annotation",
            swap_axes=True,
            )
# %%
sc.pl.dotplot(
            adata14, 
            var_names=gene_list,
            gene_symbols="feature_name",
            groupby="annotation",
            swap_axes=True,
            )
# %%
sc.pl.dotplot(
            adata17, 
            var_names=gene_list,
            gene_symbols="feature_name",
            groupby="annotation",
            swap_axes=True,
            )

# %%
from mygene import MyGeneInfo
mg = MyGeneInfo()

# Query for homologs
res = mg.querymany(gene_list, scopes='symbol', fields='homologene', species='human')

# %%
human_to_mouse = []

for gene in res:
    query = gene['query']
    if 'homologene' in gene:
        # Find all mouse genes in homologene group
        mouse_genes = [x[1] for x in gene['homologene']['genes'] if x[0] == 10090]
        for mgene in mouse_genes:
            human_to_mouse.append((query, mgene))
    else:
        human_to_mouse.append((query, None))  # No match found

# Display results
import pandas as pd
df = pd.DataFrame(human_to_mouse, columns=["Human_Gene", "Mouse_Gene"])
print(df)

#%%
# Extract the unique list of mouse gene IDs
mouse_gene_ids = list({mouse_id for _, mouse_id in human_to_mouse if mouse_id is not None})

# Query MyGene.info for gene symbols
mouse_info = mg.getgenes(mouse_gene_ids, species='mouse', fields='symbol')

# Build a mapping from NCBI Gene ID to gene symbol (safe version)
id_to_symbol = {
    str(g.get('entrezgene', g.get('_id'))): g['symbol']
    for g in mouse_info if g and 'symbol' in g
}

# Map mouse IDs to symbols in the original list
human_to_mouse_named = [
    (h, id_to_symbol.get(str(m), None)) if m is not None else (h, None)
    for h, m in human_to_mouse
]

# Convert to DataFrame
df_named = pd.DataFrame(human_to_mouse_named, columns=["Human_Gene", "Mouse_Gene_Symbol"])
print(df_named)

# %%
# drop 'H2-Ea' as they are not in the dataste
df_named = df_named[df_named["Mouse_Gene_Symbol"] != "H2-Ea"]

# %%
sc.pl.dotplot(
            adataM, 
            var_names=df_named["Mouse_Gene_Symbol"].dropna().tolist(),
            gene_symbols="symbol",
            groupby="annotation",
            swap_axes=True,
            )

# %%
# plotting gene Clec7a on umap of mouse data and annotation
sc.pl.umap(
    adataM, 
    color= ["annotation", "Cyba", "Nos1", "Nos2", "Nos3", "Xdh", "Sod1", "Sod2"],
    gene_symbols="symbol",
    frameon=False,
    size=50,
    ncols=1
)

# %%
sc.pl.dotplot(
    adataM, 
    var_names= ["Cyba", "Nos1", "Nos2", "Nos3", "Xdh", "Sod1", "Sod2", "Cat", "Gpx1", "Gpx4", "Hmox1"],
    gene_symbols="symbol",
    groupby="annotation",
    swap_axes=True,
)

# %%
