
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
# adata = adata[adata.obs["celltype"].isin(["Macrophage_I", "Macrophage_II", "HSCs", "CMP/GMP", "Monocyte_I", "Monocyte_II", "Monocyte_III", "Low_Expr"])]
adata = adata[adata.obs["celltype"].isin(["HSCs", "MEP", "Early_Ery"])]

print(adata.shape)
# (7050, 27324)

sc.pl.umap(
    adata, 
    color=['leiden','celltype'],
    legend_loc="on data",
    legend_fontsize = 6,
    show=False
)

# %%
###################################
### DE genes
###################################
columnOI='celltype'
markersMACRO = eyck.m.t.diffexp.compare_two_groups(adata, adata.obs['celltype'].isin(["Early_Ery"]), adata.obs['celltype'].isin(['HSCs', 'MEP']))
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
eyck.m.t.plot_umap(adata,diffexp['symbol'].head(8),datashader=False, cmap='coolwarm', ncol=4).display()

# %%
##################################################
##### LOAD DATA
##################################################
### Load regions
regions_name = "100k100k"
regions = chd.data.Regions(folder_dataset / "regions" / regions_name)

# %%
### Load fragments
fragments = chd.data.Fragments(
    folder_dataset / "fragments" / "100k100k"
    )
# regions.var = fragments.var

# %%
### Load motifscan
motifscan_name = "hocomocov12_1e-4"
genome_folder = pathlib.Path("/srv/data/genomes/GRCh38")
motifscan_genome = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)
motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
    path=chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / motifscan_name,
    parent=motifscan_genome,
    regions=fragments.regions,
)

# %%
### Load clustering
clustering = chd.data.Clustering.from_labels(
    adata.obs["celltype"], path=folder_dataset / "clusterings" / "cluster",
    overwrite=True
)
# %%
### Load model
model_folder = (
    chd.get_output() / "diff" / "liverfoetalH" / "100k100k" / "cluster"
)

# %%
### Load inference
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    reset=True
)
regionpositional.score(
    chd.models.diff.model.binary.Models(model_folder),
    fragments=fragments,
    clustering=clustering,
    device="cpu",
)
regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering

# %%
xs = np.concatenate([regionpositional.probs[g].mean("cluster").values for g in regionpositional.probs.keys()])
cutoff = np.exp(np.quantile(xs, 0.95))
cutoff

# %%
##################################################
##### DIFFERENTIAL SLICES
##################################################
### Filter on how accessible region is
slices = regionpositional.calculate_slices(step=25)

# # %%
# ### Filter on how big difference between the conditions is. Fc_cutoff doesn't do anything anymore?
# differential_slices = regionpositional.calculate_pairwise_differential_slices(
#     slices, fc_cutoff=2,
#     cluster_ix_a=0,
#     cluster_ix_b=2,
#     #cluster_ixs_b=[0, 1]
# )

# %%
differential_slices = regionpositional.calculate_differential_slices(
    slices, fc_cutoff=2,
)

# %%
# check if its about 5%
perc=((slices.end_position_ixs - slices.start_position_ixs) * slices.step).sum() / (regions.coordinates["end"] - regions.coordinates["start"]).sum()*100
print(round(perc,3),'%')
# 4.017%

# %%
slicescores = differential_slices.get_slice_scores(
    regions=fragments.regions, clustering=clustering
)
slicescores

# %%
slicescores.shape
# (13357, 7)

# %%
##################################################
##### ENRICHMENT in gene regions
##################################################
### Count motifs in slices
slicecounts = motifscan.count_slices(slicescores)
enrichmentTmp = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    slicescores, slicecounts
)

motifs_selected = motifscan.motifs.loc[motifscan.motifs.quality.isin(["A", "B"])]
enrichmentTmp = enrichmentTmp.query("motif in @motifs_selected.index")

# %%
### Check motif enrichment
### Top motifs are probably motifs with a lot of GC
### Usually GC regions are closed during inflammation
enrichmentTmp.loc["Early_Ery"].query("q_value < 0.05").sort_values("odds", ascending=False).head(20)

# %%
enrichmentTmp.loc["HSCs"].query("q_value < 0.05").sort_values("odds", ascending=False).head(20)

# %%
##################################################
##### ENRICHMENT in original regions
##################################################
### With 'Enrichment in gene regions' we take for each gene -10000 and +10000.
### This means certain regions are counted twice when the windows of 2 genes are overlapping.

### Calculate merged slicesscores
def map_slicescores_to_parent_regions(slicescores, regions):
    merged_slicescores = []
    slicescores2 = chd.data.regions.uncenter_multiple(slicescores, regions.coordinates)
    for cluster, clusterdata in slicescores2.groupby("cluster", observed=True):
        for chrom, chromdata in clusterdata.groupby("chrom"):
            chromdata2 = chromdata.sort_values("start_genome", ascending = True)
            chromdata2["overlap"] = (chromdata2["start_genome"].shift(-1) < chromdata2["end_genome"]).shift(1).fillna(0)
            chromdata2["overlap"] = chromdata2["overlap"].fillna(False)
            chromdata2["group"] = (~chromdata2["overlap"]).cumsum()

            chromdata3 = chromdata2.copy()[["chrom", "start_genome", "end_genome"]].rename(columns = {"start_genome":"start", "end_genome":"end"})

            chromdata3 = chromdata2.groupby("group").agg(
                chrom = ("chrom", "first"),
                start_genome = ("start_genome", "min"),
                end_genome = ("end_genome", "max")
            ).rename(columns = {"start_genome":"start", "end_genome":"end"})

            merged_slicescores.append(chromdata3.assign(cluster = cluster))
    merged_slicescores = pd.concat(merged_slicescores)
    merged_slicescores["cluster"] = pd.Categorical(merged_slicescores["cluster"], slicescores["cluster"].cat.categories)
    merged_slicescores["length"] = merged_slicescores["end"] - merged_slicescores["start"]
    merged_slicescores.index = (
        merged_slicescores["chrom"] + ":" + merged_slicescores["start"].astype(str) + "-" + merged_slicescores["end"].astype(str)
    )
    merged_slicescores.index.name = "slice"
    return merged_slicescores
merged_slicescores = map_slicescores_to_parent_regions(slicescores, fragments.regions)
merged_slicescores["start"] = merged_slicescores["start"].astype(int)
merged_slicescores["end"] = merged_slicescores["end"].astype(int)

# %%
### Count motifs in slices
merged_slicescores["region_ix"] = motifscan_genome.regions.coordinates.index.get_indexer_for(merged_slicescores["chrom"])

slicecounts = motifscan_genome.count_slices(merged_slicescores)
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    merged_slicescores, slicecounts
)

motifs_selected = motifscan.motifs.loc[motifscan.motifs.quality.isin(["A", "B"])]
enrichment = enrichment.query("motif in @motifs_selected.index")


# %%
### Check motif enrichment
enrichment.loc["Early_Ery"].query("q_value < 0.05").sort_values("odds", ascending=False).head(20)

# %%
enrichment.loc["HSCs"].query("q_value < 0.05").sort_values("odds", ascending=False).head(20)

# %%
##################################################
##### CORRECT FOR GC
##################################################
### Load genome
import pysam
import genomepy
genome = genomepy.Genome("/srv/data/genomes/GRCh38")

# %%
### Function to calculate GC
fasta = pysam.FastaFile(genome.filename)
def get_gc(fasta, slicescores):
    def extract_gc(fasta, chrom, start, end, strand):
        """
        Extract GC content in a region
        """

        actual_start = start
        if actual_start < 0:
            actual_start = 0
        seq = fasta.fetch(chrom, actual_start, end)
        if strand == -1:
            seq = seq[::-1]

        if len(seq) == 0:
            return np.nan

        # gc = np.isin(np.array(list(seq)), ["c", "g", "C", "G"]).mean()

        gc = (seq.lower().count("cg") + seq.lower().count("gc")) / len(seq)

        return gc

    gcs = []
    for chrom, start, end in tqdm.tqdm(
        zip(
            slicescores.chrom,
            slicescores.start,
            slicescores.end,
        ),
        total=len(slicescores),
    ):
        gcs.append(extract_gc(fasta, chrom, start, end, 1))
    return gcs

# %%
### Calculate GC
merged_slicescores["gc"] = get_gc(fasta, merged_slicescores)

# %%
### Check GC per condition
### Usually GC regions are closed during inflammation
merged_slicescores.groupby("cluster", observed = True)["gc"].mean()

# %%
### Run GC correction
x_smooth = np.linspace(0, 0.6, 100)
fits = pickle.load(open("/srv/data/wouters/projects/ChromatinHD_manuscript/results/1-preprocessing/lung/fits.pickle", "rb"))

# %%
fit = np.array([fit[:, 0] for k, fit in fits.items()])
expected = pd.DataFrame((np.array([np.interp(merged_slicescores["gc"], x_smooth, fit[i]) for i in range(fit.shape[0])]) * merged_slicescores["length"].values[None, :]).T, index = slicecounts.index, columns = slicecounts.columns)

# %%
### Get enrichment
enrichment = chd.models.diff.interpret.enrichment.enrichment_cluster_vs_clusters(
    merged_slicescores, slicecounts, expected = expected
)

# %%
### Add gene symbol
enrichment["symbol"] = motifscan.motifs.loc[enrichment.index.get_level_values("motif")][
    "HUMAN_gene_symbol"
].values

# %%
### Check motif enrichment after GC correction
up=enrichment.loc["Early_Ery"].query("q_value < 0.05").sort_values("odds", ascending=False)
up.head(20)

# %%
down=enrichment.loc["HSCs"].query("q_value < 0.05").sort_values("odds", ascending=False)
down.head(20)

# %%
# enrichment.loc["EC"].loc["KMT2A.H12CORE.0.P.B"]

# %%
### Save
pickle.dump(enrichment, open(folder_dataset / "enrichment.pkl", "wb"))

# %%
### Reload
enrichment = pickle.load(open(folder_dataset / "enrichment.pkl", "rb"))

# %%
##################################################
##### GROUP MOTIFS
##################################################
enrichment_grouped = chd.models.diff.interpret.enrichment.group_enrichment(
    enrichment, slicecounts, clustering.var, merge_cutoff=0.2
)

# %%
### Check motif enrichment
colsOI=['odds','p_value','q_value','log_odds','symbol','members']
upGrouped=enrichment_grouped.loc["Early_Ery"].query("q_value < 0.05").sort_values("odds", ascending=False)
upGrouped[colsOI].head(15)

# %%
downGrouped=enrichment_grouped.loc["HSCs"].query("q_value < 0.05").sort_values("odds", ascending=False)
downGrouped[colsOI].head(15)

# %%
### Check certain motif
tmp=enrichment.loc["Early_Ery"]
tmp.loc[tmp['symbol']=='GATA',["odds", "q_value"]]

# %%
# ### Search on motif family
# famName='SPI1.H12CORE.1.S.B'
# # enrichment_grouped.loc[(slice(None), famName), :]
# enrichment_grouped.loc[(slice(None), famName), 'members'].values[0]

# %%
### Get motif family of motif
tmp = enrichment_grouped.explode('members')
tmp[tmp['members'].str.contains("GATA")]

# %%
### Save
import pickle
pickle.dump(enrichment_grouped, open(folder_dataset / "enrichment_grouped.pkl", "wb"))

# %%
### Reload
enrichment_grouped = pickle.load(open(folder_dataset / "enrichment_grouped.pkl", "rb"))

# %%
# groupsToMotif = enrichment_grouped[["group", "members"]].explode("members")

# # %%
# groupsToMotif.loc[groupsToMotif['members'].str.contains('BATF')]

# %%
##################################################
##### PLOT ENRICHMENT
##################################################
# enrichment -> enrichment_oi
# enrichment_grouped -> groups_highlight (top 8 up and down) ~>(explode)~> motifs_highlight

# group_representatives = overlap between enrichment_oi and groups_highlight -> enrichment_highlight (sorted)
# => update groups_highlight


# %%
enrichment_oi = enrichment.loc["Early_Ery"]
enrichment_oi = enrichment_oi.loc[enrichment_oi["symbol"].isin(diffexp["gene"])]
enrichment_oi["logFC"] = (
    diffexp.set_index("gene").loc[enrichment_oi["symbol"]]["logFC"].values
)
enrichment_oi["FDR"] = (
    diffexp.set_index("gene").loc[enrichment_oi["symbol"]]["FDR"].values
)

### For each gene, take the motif with the highest odds
enrichment_oi = (
    enrichment_oi.sort_values("odds", ascending=False)
    .reset_index()
    .groupby("symbol", as_index=False)
    .first()
    .set_index("motif")
)

### Add color for FDR
import matplotlib.colors as mcolors
enrichment_oi['log_FDR'] = - np.log(enrichment_oi['FDR'])
norm = mpl.colors.Normalize(vmin=enrichment_oi['log_FDR'].min(), vmax=enrichment_oi['log_FDR'].max())
# cmap = mpl.cm.get_cmap('viridis')
cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", ['#00204DFF','#7C7B78FF','#FFA500'])
enrichment_oi['colorFDR'] = enrichment_oi['log_FDR'].apply(lambda x: mcolors.to_hex(cmap(norm(x))))


# %%
###################################
### Motif groups to highlight
###################################
enrichment_grouped = enrichment_grouped.sort_values(
    ["cluster", "q_value"], ascending=False
)

# %%
groups_highlight_up = (
    enrichment_grouped.query("cluster == 'Early_Ery'")
    .sort_values("log_odds", ascending=False)
    # .sort_values("q_value", ascending=True)
    .head(10)
)
warm_colors = [
    "#FF0000",
    "#FFA500",
    "#CCCC00",
    "#e6ba93",
    "#FF7F50",
    "#FFBF00",
    "#FFD700",
    "#FF00FF",
    "#800000",
    "#B7410E",
]
groups_highlight_up["color"] = warm_colors[: len(groups_highlight_up)]

# %%
groups_highlight_down = (
    enrichment_grouped.query("cluster == 'Early_Ery'")
    .sort_values("log_odds", ascending=True)
    # .sort_values("q_value", ascending=True)
    .head(10)
)
cold_colors = [
    "#0000FF",
    "#00FFFF",
    "#008080",
    "#00FF00",
    "#008000",
    "#00FF7F",
    "#00FFD4",
    "#008B8B",
    "#000080",
    "#0000CD",
]
groups_highlight_down["color"] = cold_colors[: len(groups_highlight_down)]
groups_highlight = pd.concat(
    [
        groups_highlight_up,
        groups_highlight_down,
    ]
)



# %%
###################################
### Prepare plot
###################################
motifs_highlight = groups_highlight[["group", "members", "color"]].explode("members")
motifs_highlight

# %%
motifs_highlight = motifs_highlight[~motifs_highlight.duplicated(subset=['members'])]

# %%
enrichment_oi["color"] = (
    enrichment_oi[[]]
    .join(motifs_highlight.set_index("members")["color"])["color"]
    .fillna("lightgray")
)
enrichment_oi["group"] = enrichment_oi[[]].join(
    motifs_highlight.set_index("members")["group"]
)["group"]

# %%
# remove groups with no representative TF
enrichment_oi["abs_logFC"] = np.abs(enrichment_oi["logFC"])
group_representatives = (
    enrichment_oi.dropna(subset=["group"])
    .sort_values("abs_logFC", ascending=False)
    .reset_index()
    .groupby("group")
    .first()
)

# %%
groups_highlight = groups_highlight.loc[groups_highlight.index.get_level_values("motif").isin(group_representatives.index)].copy()

groups_highlight["label"] = group_representatives.loc[
    groups_highlight.index.get_level_values("motif")
]["symbol"].values

enrichment_highlight = group_representatives
enrichment_highlight.sort_values("odds").groupby("group")["odds"].max()

# %%
### Per group, the motif that represents the group
enrichment_highlight
### For each group, see all the members
groups_highlight



# %%
###################################
### Make plot - color per group
###################################
fig = pp.grid.Figure()
ax = fig.main.add(pp.grid.Panel((3,3)))
alpha = 0.4 + (enrichment_oi.index.isin(enrichment_highlight["motif"]) * 0.6)

points = ax.scatter(
    enrichment_oi["logFC"],
    enrichment_oi["log_odds"],
    c=enrichment_oi["color"],
    s=30,
    clip_on=False,
    zorder=10,
    alpha=alpha,
    lw=0,
)
texts = []
for i, row in enrichment_highlight.iterrows():
    x, y = row[["logFC", "log_odds"]]
    text = ax.annotate(
        row["symbol"],
        (x, y),
        fontsize=10,
        color=row["color"],
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", color=row["color"]),
        bbox=dict(pad=-5, facecolor="none", edgecolor="none"),
        zorder=20,
    )

    text.set_path_effects([mpl.patheffects.withStroke(linewidth=2, foreground="white")])
    texts.append(text)

### axis in middle
ax.axhline(0, color="black", linewidth=1)
ax.axvline(0, color="black", linewidth=1)
xmax = max(abs(enrichment_oi["logFC"].min()), enrichment_oi["logFC"].max())
ymax = max(abs(enrichment_oi["log_odds"].min()), enrichment_oi["log_odds"].max())

### spines
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

### ticks
# xticks = np.log2([ 1 / 4, 1, 4])
# yticks = np.log2([1 / 2, 1, 2])
# ax.set_xticks(xticks)
# ax.set_yticks(yticks)
# ax.set_xticklabels(["1/4", "1", "4"], fontsize=5)
# ax.set_yticklabels(["1/2", "1", "2"], fontsize=5)
ax.set_xlabel("TF mRNA fold-change", fontsize=7)
ax.set_ylabel("TFBS\nodds-ratio", ha="right", va="center", fontsize=7)

### labels
import adjustText
fig.plot()
adjustText.adjust_text(texts, ax=ax, ensure_inside_axes=False)

### Save plot
fig.display()
# fig.savefig(saveFolder / "enrichmentPlots" / "enrichment.png", dpi=600, bbox_inches="tight")

# %%
