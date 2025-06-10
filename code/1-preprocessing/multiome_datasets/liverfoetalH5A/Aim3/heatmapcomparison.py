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
dataset_name = "liverfoetalH"
adataSN = pickle.load((folder_data / dataset_name / "adata_sn_kc_annotated.pkl").open("rb"))
adataSN = adataSN[adataSN.obs["annotation"].isin(["Monocyte", "Macrophage"])]


# %%
dataset_name = "mouse_ALK1KO"
adataM = pickle.load((folder_data / dataset_name / "adata_mouse_kc_annotated.pkl").open("rb"))

# %%
dataset_name = "suo22_kc"
adata8 = pickle.load((folder_data / dataset_name / "adata_sc_8_annotated.pkl").open("rb"))
adata11 = pickle.load((folder_data / dataset_name / "adata_sc_11_annotated.pkl").open("rb"))
adata14 = pickle.load((folder_data / dataset_name / "adata_sc_14_annotated.pkl").open("rb"))
adata17 = pickle.load((folder_data / dataset_name / "adata_sc_17_annotated.pkl").open("rb"))

# %% [markdown]
###############################################################################################
# # heatmap of top 100 DEG genes in multiome
###############################################################################################

#%%
sc.tl.rank_genes_groups(
    adataSN, groupby="annotation", method="wilcoxon", key_added="wilcoxon", use_raw=False,
)

# %%
# visualize results using dotplot
sc.pl.rank_genes_groups_dotplot(adataSN, 
                                key="wilcoxon", 
                                gene_symbols="symbol",
                                n_genes=20)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adataSN.uns["wilcoxon"]["names"][cluster] for cluster in adataSN.uns["wilcoxon"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 100 # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
top_genes = pd.Series(top_genes_df.T.values.flatten()).dropna()


# %%
sc.pl.heatmap(adata14, 
              var_names=top_genes,
              groupby="annotation",
              cmap="Reds",
              standard_scale='var',
              )


 # %%
sc.pl.heatmap(adataSN, 
              var_names=top_genes,
              groupby="annotation",
              layer='magic',
              cmap="Reds",
              #standard_scale='var',
              )


# %% [markdown]
###############################################################################################
# # heatmap of top 100 DEG genes in scRNAseq
###############################################################################################

# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]


def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene_ids").loc[gene, "symbol"]


#%%
sc.tl.rank_genes_groups(
    adata14, groupby="annotation", method="wilcoxon", key_added="wilcoxon", use_raw=False,
)

# %%
# visualize results using dotplot
sc.pl.rank_genes_groups_dotplot(adata14, 
                                key="wilcoxon", 
                                gene_symbols="feature_name",
                                n_genes=10)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adata14.uns["wilcoxon"]["names"][cluster] for cluster in adata14.uns["wilcoxon"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 100  # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
top_genes = pd.Series(top_genes_df.T.values.flatten()).dropna()
top_genes = top_genes[~top_genes.isin([ # genes not present in multiome
    'ENSG00000155659'  
    ])]

# %%
sc.pl.heatmap(adata14, 
              var_names=top_genes,
              groupby="annotation",
              cmap="Reds",
              standard_scale='var',
              )

# %%
sc.pl.heatmap(adataSN, 
              var_names=top_genes,
              groupby="annotation",
              layer='magic',
              cmap="Reds",
              standard_scale='var',
              )

# %% [markdown]
###############################################################################################
# # heatmap of top 100 DEG genes from human multiome to mouse
###############################################################################################

#%%
sc.tl.rank_genes_groups(
    adataSN, groupby="annotation", method="wilcoxon", key_added="wilcoxon", use_raw=False,
)

# %%
# visualize results using dotplot
sc.pl.rank_genes_groups_dotplot(adataSN, 
                                key="wilcoxon", 
                                gene_symbols="symbol",
                                n_genes=20)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adataSN.uns["wilcoxon"]["names"][cluster] for cluster in adataSN.uns["wilcoxon"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 100 # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
top_genes = pd.Series(top_genes_df.T.values.flatten()).dropna()


# %%
from mygene import MyGeneInfo
mg = MyGeneInfo()

def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene_ids").loc[gene, "symbol"]
top_genes_symbols = symbol(adataSN, top_genes)

# Query for homologs
res = mg.querymany(top_genes_symbols, scopes='symbol', fields='homologene', species='human')

# %%
human_to_mouse = []

for gene in res:
    query = gene['query']
    if 'homologene' in gene:
        # Find all mouse genes in homologene group
        mouse_genes = [x[1] for x in gene['homologene']['genes'] if x[0] == 10090]
        if mouse_genes: # Check if there are any mouse genes found
            # Append only the first mouse gene's Entrez ID
            human_to_mouse.append((query, mouse_genes[0])) 
        else:
            human_to_mouse.append((query, None))
    else:
        human_to_mouse.append((query, None))  # No match found

# Display results
import pandas as pd
df = pd.DataFrame(human_to_mouse, columns=["Human_Gene", "Mouse_Gene"])
print(df)

# %%
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
# drop genes as they are not in the dataste
df_named = df_named[~df_named["Mouse_Gene_Symbol"].isin(["Csta1", "Gapdh-ps15", "H2-Ea", "Hbb-y",'S100a11-ps','Uba52-ps1' ])]
df_named = df_named.dropna()
sc.pl.heatmap(adataM, 
              var_names=df_named["Mouse_Gene_Symbol"].dropna().tolist(), 
              gene_symbols="symbol",
              groupby="annotation",
              cmap="Reds",
              standard_scale='var',
              )


# %%
sc.pl.heatmap(adataSN, 
              var_names=df_named["Human_Gene"].dropna().tolist(), 
              gene_symbols="symbol",
              groupby="annotation",
              cmap="Reds",
              standard_scale='var',
              layer='magic',
              )

# %% [markdown]
###############################################################################################
# # heatmap of top 100 DEG genes from human scRNA to mouse
###############################################################################################

#%%
sc.tl.rank_genes_groups(
    adata14, groupby="annotation", method="wilcoxon", key_added="wilcoxon", use_raw=False,
)

# %%
# visualize results using dotplot
sc.pl.rank_genes_groups_dotplot(adata14, 
                                key="wilcoxon", 
                                gene_symbols="feature_name",
                                n_genes=10)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adata14.uns["wilcoxon"]["names"][cluster] for cluster in adata14.uns["wilcoxon"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 100  # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})
top_genes = pd.Series(top_genes_df.T.values.flatten()).dropna()


# %%
from mygene import MyGeneInfo
mg = MyGeneInfo()

def symbol(adata, gene):
    return adata.var.loc[gene, "feature_name"]
top_genes_symbols = symbol(adata14, top_genes)

# Query for homologs
res = mg.querymany(top_genes_symbols, scopes='symbol', fields='homologene', species='human')

# %%
human_to_mouse = []

for gene in res:
    query = gene['query']
    if 'homologene' in gene:
        # Find all mouse genes in homologene group
        mouse_genes = [x[1] for x in gene['homologene']['genes'] if x[0] == 10090]
        if mouse_genes: # Check if there are any mouse genes found
            # Append only the first mouse gene's Entrez ID
            human_to_mouse.append((query, mouse_genes[0])) 
        else:
            human_to_mouse.append((query, None))
    else:
        human_to_mouse.append((query, None))  # No match found

# Display results
import pandas as pd
df = pd.DataFrame(human_to_mouse, columns=["Human_Gene", "Mouse_Gene"])
print(df)

# %%
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
# drop genes as they are not in the dataste
df_named = df_named[~df_named["Mouse_Gene_Symbol"].isin(["Csta1", "Gapdh-ps15", "H2-Ea", "Hbb-y",'S100a11-ps','Uba52-ps1' ])]
df_named = df_named.dropna()
sc.pl.heatmap(adataM, 
              var_names=df_named["Mouse_Gene_Symbol"].dropna().tolist(), 
              gene_symbols="symbol",
              groupby="annotation",
              cmap="Reds",
              standard_scale='var',
              )


# %%
sc.pl.heatmap(adata14, 
              var_names=df_named["Human_Gene"].dropna().tolist(), 
              gene_symbols="feature_name",
              groupby="annotation",
              cmap="Reds",
              standard_scale='var',
              )

# %%
