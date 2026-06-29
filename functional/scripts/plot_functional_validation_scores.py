# Panels: Supplementary Figure 2(c,d,e)


# Monorepo path constants (monorepo relative paths)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PRECOMPUTED = PROJECT_ROOT / "data" / "precomputed"
TF_FAMILY_PATH = PROJECT_ROOT / "data" / "TF_family" / "TF_Information.txt"

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, hypergeom, gaussian_kde, wilcoxon


pred_path = DATA_PRECOMPUTED / "2_1_functional_classifier_results" / "predictions" / "esm_window_site_pdb_5_folds_ensemble_predictions.csv"
pos_path = PROJECT_ROOT / "data" / "dataset_phos_site" / "TF_positive_phos_site_0608.csv"
cluster_path = PROJECT_ROOT / "data" / "dataset_phos_site" / "co_working_multi_site_with_PMID.csv"

out_dir = PROJECT_ROOT / "results" / "2_1_functional_classifier_results" / "single_model_rank_eval_5_folds_ensemble"
out_dir.mkdir(parents=True, exist_ok=True)

score_col = "mean_prob"
ks = [1, 3, 5, 10]
top_percentiles = [0.01, 0.05, 0.10, 0.20]
n_permutations = 10000
random_seed = 43
exclude_known_positive_from_cluster = True
cluster_gap_for_auto_id = 30
allow_relaxed_background = True

FONT_SIZE_SHIFT = 4

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "font.size": 10 + FONT_SIZE_SHIFT,
    "axes.labelsize": 11 + FONT_SIZE_SHIFT,
    "axes.titlesize": 12 + FONT_SIZE_SHIFT,
    "xtick.labelsize": 10 + FONT_SIZE_SHIFT,
    "ytick.labelsize": 10 + FONT_SIZE_SHIFT,
    "legend.fontsize": 9 + FONT_SIZE_SHIFT,
})


def build_site_key(df):
    df = df.copy()
    df["ACC_ID"] = df["ACC_ID"].astype(str).str.strip()
    df["POSITION"] = pd.to_numeric(df["POSITION"], errors="coerce").astype("Int64")

    if "RESIDUE" in df.columns:
        residue = df["RESIDUE"].astype(str).str.extract(r"([STYsty])", expand=False)
    elif "MOD_RSD" in df.columns:
        residue = df["MOD_RSD"].astype(str).str.extract(r"([STYsty])", expand=False)
    elif "INDEX" in df.columns:
        residue = df["INDEX"].astype(str).str.extract(r"_([STYsty])\d+", expand=False)
    else:
        raise ValueError("No residue column found.")

    residue = residue.astype(str).str.upper()
    df["RESIDUE_KEY"] = residue
    df["SITE_KEY"] = df["ACC_ID"] + "_" + df["RESIDUE_KEY"] + df["POSITION"].astype(str)
    return df


def random_mrr_expectation(n, p):
    if n <= 0 or p <= 0 or p > n:
        return np.nan

    value = 0.0
    prob_no_positive_before = 1.0

    for r in range(1, n - p + 2):
        prob_first_at_r = prob_no_positive_before * p / (n - r + 1)
        value += prob_first_at_r / r

        if r < n - p + 1:
            prob_no_positive_before *= (n - p - (r - 1)) / (n - (r - 1))

    return value


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    if n == 0:
        return np.array([], dtype=float)

    valid = np.isfinite(p_values)
    q_values = np.full(n, np.nan, dtype=float)

    if valid.sum() == 0:
        return q_values

    valid_values = p_values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]

    adjusted = ranked * len(valid_values) / (np.arange(len(valid_values)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    valid_q = np.empty(len(valid_values), dtype=float)
    valid_q[order] = adjusted
    q_values[valid] = valid_q
    return q_values


def p_to_star(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_value(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.0, length=4)
    ax.grid(False)


def run_permutation_tests(per_protein, ks, n_permutations, random_seed):
    rng = np.random.default_rng(random_seed)

    protein_info = per_protein[["ACC_ID", "candidate_n", "positive_n"]].copy()
    total_positive = protein_info["positive_n"].sum()
    protein_n = len(protein_info)

    macro_null = {k: np.zeros(n_permutations, dtype=float) for k in ks}
    micro_null = {k: np.zeros(n_permutations, dtype=float) for k in ks}
    mrr_null = np.zeros(n_permutations, dtype=float)

    for b in range(n_permutations):
        macro_sum = {k: 0.0 for k in ks}
        micro_hits = {k: 0 for k in ks}
        rr_sum = 0.0

        for _, row in protein_info.iterrows():
            n = int(row["candidate_n"])
            p = int(row["positive_n"])
            random_ranks = rng.choice(np.arange(1, n + 1), size=p, replace=False)

            first_rank = int(random_ranks.min())
            rr_sum += 1.0 / first_rank

            for k in ks:
                real_k = min(k, n)
                hits = int((random_ranks <= real_k).sum())
                macro_sum[k] += hits / p
                micro_hits[k] += hits

        for k in ks:
            macro_null[k][b] = macro_sum[k] / protein_n
            micro_null[k][b] = micro_hits[k] / total_positive

        mrr_null[b] = rr_sum / protein_n

    return macro_null, micro_null, mrr_null


def save_figure(fig, base_path):
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def safe_kde(values, x_grid):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.zeros_like(x_grid)

    if len(np.unique(values)) == 1:
        center = float(values[0])
        width = 0.03
        density = np.exp(-0.5 * ((x_grid - center) / width) ** 2)
        density /= np.trapz(density, x_grid)
        return density

    kde = gaussian_kde(values)
    return kde(x_grid)


def safe_string(value):
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    if text == "":
        text = "NA"
    return text


def assign_cluster_ids(cluster_df, gap=30):
    df = cluster_df.copy()

    manual_cols = [
        "CLUSTER_ID",
        "Cluster_ID",
        "cluster_id",
        "CLUSTER",
        "Cluster",
        "GROUP_ID",
        "Group_ID",
        "REGION_ID",
        "Region_ID",
    ]

    existing_col = None
    for col in manual_cols:
        if col in df.columns:
            non_empty = df[col].astype(str).str.strip().replace({"nan": "", "None": ""})
            if (non_empty != "").any():
                existing_col = col
                break

    df["_AUTO_CLUSTER_ID"] = ""

    ref_col = None
    for col in ["PMID", "Literature_title", "TITLE", "Reference", "SOURCE", "Source"]:
        if col in df.columns:
            ref_col = col
            break

    group_cols = ["ACC_ID"]
    if ref_col is not None:
        group_cols.append(ref_col)

    auto_ids = []
    for group_key, group in df.sort_values(["ACC_ID", "POSITION"]).groupby(group_cols, dropna=False):
        group = group.sort_values("POSITION").copy()

        if isinstance(group_key, tuple):
            acc_id = safe_string(group_key[0])
            ref_value = safe_string(group_key[1])
        else:
            acc_id = safe_string(group_key)
            ref_value = "NA"

        cluster_idx = 1
        last_pos = None

        for idx, row in group.iterrows():
            pos = int(row["POSITION"])
            if last_pos is not None and abs(pos - last_pos) > gap:
                cluster_idx += 1

            auto_id = f"{acc_id}_{ref_value}_C{cluster_idx:02d}"
            auto_ids.append((idx, auto_id))
            last_pos = pos

    auto_map = dict(auto_ids)
    df["_AUTO_CLUSTER_ID"] = df.index.map(auto_map)

    if existing_col is not None:
        manual = df[existing_col].astype(str).str.strip()
        manual = manual.replace({"nan": "", "None": ""})
        df["CLUSTER_ID"] = manual
        empty_mask = df["CLUSTER_ID"] == ""
        df.loc[empty_mask, "CLUSTER_ID"] = df.loc[empty_mask, "_AUTO_CLUSTER_ID"]
    else:
        df["CLUSTER_ID"] = df["_AUTO_CLUSTER_ID"]

    df = df.drop(columns=["_AUTO_CLUSTER_ID"])
    return df


def empirical_greater_p(null_values, observed):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]

    if len(null_values) == 0 or not np.isfinite(observed):
        return np.nan

    return (np.sum(null_values >= observed) + 1) / (len(null_values) + 1)


def empirical_less_p(null_values, observed):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]

    if len(null_values) == 0 or not np.isfinite(observed):
        return np.nan

    return (np.sum(null_values <= observed) + 1) / (len(null_values) + 1)


def safe_wilcoxon_greater(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    if np.allclose(x - y, 0):
        return np.nan

    try:
        return wilcoxon(x, y, alternative="greater").pvalue
    except ValueError:
        return np.nan


def select_background_pool(protein_df, current_cluster_sites, known_positive_sites, all_cluster_sites, allow_relaxed=True):
    current_cluster_sites = set(current_cluster_sites)

    strict_exclude = set(known_positive_sites) | set(all_cluster_sites)
    strict_pool = protein_df[~protein_df["SITE_KEY"].isin(strict_exclude)].copy()
    if len(strict_pool) > 0:
        return strict_pool, "strict"

    if allow_relaxed:
        relaxed_exclude = set(known_positive_sites) | current_cluster_sites
        relaxed_pool = protein_df[~protein_df["SITE_KEY"].isin(relaxed_exclude)].copy()
        if len(relaxed_pool) > 0:
            return relaxed_pool, "relaxed_without_other_clusters"

        minimal_pool = protein_df[~protein_df["SITE_KEY"].isin(current_cluster_sites)].copy()
        if len(minimal_pool) > 0:
            return minimal_pool, "minimal_without_current_cluster"

    return protein_df.iloc[0:0].copy(), "unavailable"


def build_pred_ranked(pred, score_col):
    pred_ranked = pred.copy()
    pred_ranked = pred_ranked.sort_values(score_col, ascending=False).reset_index(drop=True)
    pred_ranked["global_rank"] = np.arange(1, len(pred_ranked) + 1)
    pred_ranked["global_percentile"] = pred_ranked["global_rank"] / len(pred_ranked)
    pred_ranked = pred_ranked.sort_values(["ACC_ID", score_col], ascending=[True, False]).copy()
    pred_ranked["rank_within_protein"] = pred_ranked.groupby("ACC_ID").cumcount() + 1
    pred_ranked["candidate_n_within_protein"] = pred_ranked.groupby("ACC_ID")["SITE_KEY"].transform("size")
    return pred_ranked


def protein_cluster_metrics_at_pct(protein_df, cluster_sites, pct):
    protein_df = protein_df.sort_values("rank_within_protein").copy()
    candidate_n = len(protein_df)
    cluster_sites = set(cluster_sites)
    cluster_n = len(cluster_sites)

    if candidate_n == 0 or cluster_n == 0:
        return None

    top_n = max(1, int(np.ceil(candidate_n * pct)))
    is_cluster = protein_df["SITE_KEY"].isin(cluster_sites).to_numpy()
    ranks = protein_df["rank_within_protein"].to_numpy()
    is_top = ranks <= top_n

    a = int((is_top & is_cluster).sum())
    b = int((is_top & ~is_cluster).sum())
    c = int((~is_top & is_cluster).sum())
    d = int((~is_top & ~is_cluster).sum())

    table = [[a, b], [c, d]]
    _, fisher_p = fisher_exact(table, alternative="greater")

    top_cluster_rate = a / top_n if top_n > 0 else np.nan
    overall_cluster_rate = cluster_n / candidate_n if candidate_n > 0 else np.nan
    fold_enrichment = (
        top_cluster_rate / overall_cluster_rate if overall_cluster_rate > 0 else np.nan
    )
    recall_in_top = a / cluster_n if cluster_n > 0 else np.nan
    random_recall = top_n / candidate_n if candidate_n > 0 else np.nan
    hypergeom_p = hypergeom.sf(a - 1, candidate_n, cluster_n, top_n)

    return {
        "top_n": top_n,
        "recall_in_top": recall_in_top,
        "random_recall": random_recall,
        "fold_enrichment": fold_enrichment,
        "fisher_p": fisher_p,
        "hypergeom_p": hypergeom_p,
    }


def run_protein_cluster_fold_permutation(pred_ranked, cluster_by_acc, top_percentiles, n_permutations, random_seed):
    """Permutation test on macro recall excess within proteins (recall_in_top - random_recall)."""
    rng = np.random.default_rng(random_seed)
    eval_acc_ids = sorted(cluster_by_acc.keys())

    protein_dfs = {
        acc_id: pred_ranked[pred_ranked["ACC_ID"].astype(str) == acc_id].copy()
        for acc_id in eval_acc_ids
    }

    permutation_rows = []

    for pct in top_percentiles:
        observed_excesses = []
        for acc_id in eval_acc_ids:
            metrics = protein_cluster_metrics_at_pct(protein_dfs[acc_id], cluster_by_acc[acc_id], pct)
            observed_excesses.append(metrics["recall_in_top"] - metrics["random_recall"])

        observed_macro_excess = float(np.mean(observed_excesses))
        null_macro_excess = np.zeros(n_permutations, dtype=float)

        for b in range(n_permutations):
            null_excesses = []
            for acc_id in eval_acc_ids:
                protein_df = protein_dfs[acc_id]
                candidate_n = len(protein_df)
                cluster_n = len(cluster_by_acc[acc_id])

                if cluster_n >= candidate_n:
                    null_excesses.append(0.0)
                    continue

                random_idx = rng.choice(candidate_n, size=cluster_n, replace=False)
                random_cluster_sites = set(protein_df.iloc[random_idx]["SITE_KEY"])
                metrics = protein_cluster_metrics_at_pct(protein_df, random_cluster_sites, pct)
                null_excesses.append(metrics["recall_in_top"] - metrics["random_recall"])

            null_macro_excess[b] = float(np.mean(null_excesses))

        perm_p = (np.sum(null_macro_excess >= observed_macro_excess) + 1) / (n_permutations + 1)
        ci_low, ci_high = np.percentile(null_macro_excess, [2.5, 97.5])

        permutation_rows.append({
            "top_percentile": pct,
            "observed_macro_recall_excess": observed_macro_excess,
            "null_macro_recall_excess_mean": float(np.mean(null_macro_excess)),
            "null_macro_recall_excess_ci_lower": float(ci_low),
            "null_macro_recall_excess_ci_upper": float(ci_high),
            "permutation_p_greater": float(perm_p),
        })

    return pd.DataFrame(permutation_rows)


def compute_protein_level_cluster_summary(matched_cluster, pred_ranked, top_percentiles):
    cluster_by_acc = {
        str(acc_id): set(group["SITE_KEY"])
        for acc_id, group in matched_cluster.groupby("ACC_ID")
    }

    protein_rows = []

    for acc_id, protein_df in pred_ranked.groupby("ACC_ID"):
        acc_id = str(acc_id)
        cluster_sites = cluster_by_acc.get(acc_id)
        if not cluster_sites:
            continue

        protein_df = protein_df.sort_values("rank_within_protein").copy()
        candidate_n = len(protein_df)
        cluster_n = len(cluster_sites)

        row = {
            "ACC_ID": acc_id,
            "candidate_n": candidate_n,
            "cluster_n": cluster_n,
        }

        if "GENE" in matched_cluster.columns:
            genes = matched_cluster.loc[matched_cluster["ACC_ID"].astype(str) == acc_id, "GENE"].astype(str).unique()
            row["GENE"] = ";".join(sorted(genes))

        for pct in top_percentiles:
            metrics = protein_cluster_metrics_at_pct(protein_df, cluster_sites, pct)
            if metrics is None:
                continue

            pct_key = f"pct_{int(pct * 100):02d}"
            row[f"top_n_{pct_key}"] = metrics["top_n"]
            row[f"recall_in_top_{pct_key}"] = metrics["recall_in_top"]
            row[f"random_recall_{pct_key}"] = metrics["random_recall"]
            row[f"fold_enrichment_{pct_key}"] = metrics["fold_enrichment"]
            row[f"fisher_p_{pct_key}"] = metrics["fisher_p"]
            row[f"hypergeom_p_{pct_key}"] = metrics["hypergeom_p"]

        protein_rows.append(row)

    per_protein_cluster = pd.DataFrame(protein_rows)
    if per_protein_cluster.empty:
        raise ValueError("No protein-level cluster metrics could be computed.")

    summary_rows = []
    for pct in top_percentiles:
        pct_key = f"pct_{int(pct * 100):02d}"
        recall_col = f"recall_in_top_{pct_key}"
        random_col = f"random_recall_{pct_key}"
        fold_col = f"fold_enrichment_{pct_key}"
        fisher_col = f"fisher_p_{pct_key}"
        hyper_col = f"hypergeom_p_{pct_key}"

        fisher_q = benjamini_hochberg(per_protein_cluster[fisher_col].astype(float))
        hyper_q = benjamini_hochberg(per_protein_cluster[hyper_col].astype(float))

        summary_rows.append({
            "top_percentile": pct,
            "top_label": f"Top {int(pct * 100)}%",
            "evaluated_protein_n": len(per_protein_cluster),
            "macro_recall_in_top": float(per_protein_cluster[recall_col].mean()),
            "macro_random_recall": float(per_protein_cluster[random_col].mean()),
            "macro_fold_enrichment": float(per_protein_cluster[fold_col].mean()),
            "recall_in_top": float(per_protein_cluster[recall_col].mean()),
            "random_recall": float(per_protein_cluster[random_col].mean()),
            "fold_enrichment": float(per_protein_cluster[fold_col].mean()),
            "median_recall_in_top": float(per_protein_cluster[recall_col].median()),
            "protein_fisher_q_bh_median": float(np.median(fisher_q)),
            "protein_hypergeom_q_bh_median": float(np.median(hyper_q)),
            "fisher_p_greater": float(np.median(per_protein_cluster[fisher_col])),
            "hypergeom_p_greater": float(np.median(per_protein_cluster[hyper_col])),
        })

    protein_cluster_summary = pd.DataFrame(summary_rows)
    protein_cluster_summary["fisher_q_bh"] = benjamini_hochberg(
        protein_cluster_summary["fisher_p_greater"]
    )
    protein_cluster_summary["hypergeom_q_bh"] = benjamini_hochberg(
        protein_cluster_summary["hypergeom_p_greater"]
    )
    protein_cluster_summary["fisher_star"] = protein_cluster_summary["fisher_q_bh"].apply(p_to_star)

    return per_protein_cluster, protein_cluster_summary, cluster_by_acc


pred = pd.read_csv(pred_path, low_memory=False)
pos = pd.read_csv(pos_path, low_memory=False)
cluster = pd.read_csv(cluster_path, low_memory=False)

if score_col not in pred.columns:
    raise ValueError(f"{score_col} is not in prediction table.")

pred = build_site_key(pred)
pos = build_site_key(pos)
cluster = build_site_key(cluster)

pred[score_col] = pd.to_numeric(pred[score_col], errors="coerce")

pred = pred.dropna(subset=["ACC_ID", "POSITION", "RESIDUE_KEY", "SITE_KEY", score_col]).copy()
pos = pos.dropna(subset=["ACC_ID", "POSITION", "RESIDUE_KEY", "SITE_KEY"]).copy()
cluster = cluster.dropna(subset=["ACC_ID", "POSITION", "RESIDUE_KEY", "SITE_KEY"]).copy()

pred = pred.sort_values(score_col, ascending=False)
pred = pred.drop_duplicates("SITE_KEY", keep="first")

pos = pos.drop_duplicates("SITE_KEY", keep="first")
cluster = cluster.drop_duplicates("SITE_KEY", keep="first")

pred_site_set = set(pred["SITE_KEY"])

pos["in_prediction_table"] = pos["SITE_KEY"].isin(pred_site_set)
matched_pos = pos[pos["in_prediction_table"]].copy()
missing_pos = pos[~pos["in_prediction_table"]].copy()

eval_acc_ids = sorted(matched_pos["ACC_ID"].unique())

rank_df = pred[pred["ACC_ID"].isin(eval_acc_ids)].copy()
rank_df["is_positive"] = rank_df["SITE_KEY"].isin(set(matched_pos["SITE_KEY"]))
rank_df = rank_df.sort_values(["ACC_ID", score_col], ascending=[True, False])
rank_df["rank"] = rank_df.groupby("ACC_ID").cumcount() + 1

protein_rows = []

for acc_id, g in rank_df.groupby("ACC_ID"):
    g = g.sort_values("rank")

    positive_ranks = g.loc[g["is_positive"], "rank"].to_numpy()
    candidate_n = len(g)
    positive_n = len(positive_ranks)

    if positive_n == 0:
        continue

    row = {
        "ACC_ID": acc_id,
        "candidate_n": candidate_n,
        "positive_n": positive_n,
        "first_positive_rank": int(positive_ranks.min()),
        "rr": 1.0 / positive_ranks.min(),
        "random_mrr": random_mrr_expectation(candidate_n, positive_n),
    }

    if "GENE" in matched_pos.columns:
        genes = matched_pos.loc[matched_pos["ACC_ID"] == acc_id, "GENE"].astype(str).unique()
        row["GENE"] = ";".join(sorted(genes))

    for k in ks:
        real_k = min(k, candidate_n)
        hits = int((positive_ranks <= real_k).sum())
        recall = hits / positive_n
        random_recall = real_k / candidate_n

        row[f"hits_at_{k}"] = hits
        row[f"recall_at_{k}"] = recall
        row[f"random_recall_at_{k}"] = random_recall
        row[f"fold_over_random_at_{k}"] = recall / random_recall if random_recall > 0 else np.nan

    protein_rows.append(row)

per_protein = pd.DataFrame(protein_rows)

summary_rows = []

for k in ks:
    hits_col = f"hits_at_{k}"
    recall_col = f"recall_at_{k}"
    random_col = f"random_recall_at_{k}"

    macro_model = per_protein[recall_col].mean()
    micro_model = per_protein[hits_col].sum() / per_protein["positive_n"].sum()

    macro_random = per_protein[random_col].mean()
    micro_random = (per_protein["positive_n"] * per_protein[random_col]).sum() / per_protein["positive_n"].sum()

    summary_rows.append({
        "K": k,
        "macro_recall_model": macro_model,
        "micro_recall_model": micro_model,
        "macro_recall_random": macro_random,
        "micro_recall_random": micro_random,
        "macro_fold_over_random": macro_model / macro_random if macro_random > 0 else np.nan,
        "micro_fold_over_random": micro_model / micro_random if micro_random > 0 else np.nan,
    })

summary = pd.DataFrame(summary_rows)

mrr_summary = pd.DataFrame([{
    "mrr_model": per_protein["rr"].mean(),
    "mrr_random": per_protein["random_mrr"].mean(),
    "mrr_fold_over_random": per_protein["rr"].mean() / per_protein["random_mrr"].mean(),
    "evaluated_protein_n": len(per_protein),
    "matched_positive_site_n": int(per_protein["positive_n"].sum()),
    "missing_positive_site_n": len(missing_pos),
}])

fisher_rows = []

for k in ks:
    is_topk = rank_df["rank"] <= k
    is_positive = rank_df["is_positive"]

    a = int((is_topk & is_positive).sum())
    b = int((is_topk & ~is_positive).sum())
    c = int((~is_topk & is_positive).sum())
    d = int((~is_topk & ~is_positive).sum())

    table = [[a, b], [c, d]]
    odds_ratio, fisher_p = fisher_exact(table, alternative="greater")

    total_sites = a + b + c + d
    total_positive = a + c
    total_topk = a + b

    topk_positive_rate = a / total_topk if total_topk > 0 else np.nan
    overall_positive_rate = total_positive / total_sites if total_sites > 0 else np.nan
    fold_enrichment = topk_positive_rate / overall_positive_rate if overall_positive_rate > 0 else np.nan

    hypergeom_p = hypergeom.sf(a - 1, total_sites, total_positive, total_topk)

    fisher_rows.append({
        "K": k,
        "topk_positive": a,
        "topk_non_positive": b,
        "remaining_positive": c,
        "remaining_non_positive": d,
        "total_topk": total_topk,
        "total_sites": total_sites,
        "total_positive": total_positive,
        "topk_positive_rate": topk_positive_rate,
        "overall_positive_rate": overall_positive_rate,
        "fold_enrichment": fold_enrichment,
        "odds_ratio": odds_ratio,
        "fisher_p_greater": fisher_p,
        "hypergeom_p_greater": hypergeom_p,
    })

fisher_summary = pd.DataFrame(fisher_rows)
fisher_summary["fisher_q_bh"] = benjamini_hochberg(fisher_summary["fisher_p_greater"])
fisher_summary["hypergeom_q_bh"] = benjamini_hochberg(fisher_summary["hypergeom_p_greater"])
fisher_summary["fisher_star"] = fisher_summary["fisher_q_bh"].apply(p_to_star)

macro_null, micro_null, mrr_null = run_permutation_tests(
    per_protein=per_protein,
    ks=ks,
    n_permutations=n_permutations,
    random_seed=random_seed,
)

permutation_rows = []

for k in ks:
    observed_macro = float(summary.loc[summary["K"] == k, "macro_recall_model"].iloc[0])
    observed_micro = float(summary.loc[summary["K"] == k, "micro_recall_model"].iloc[0])

    macro_p = (np.sum(macro_null[k] >= observed_macro) + 1) / (n_permutations + 1)
    micro_p = (np.sum(micro_null[k] >= observed_micro) + 1) / (n_permutations + 1)

    macro_ci_low, macro_ci_high = np.percentile(macro_null[k], [2.5, 97.5])
    micro_ci_low, micro_ci_high = np.percentile(micro_null[k], [2.5, 97.5])

    permutation_rows.append({
        "metric": f"Macro Recall@{k}",
        "K": k,
        "observed": observed_macro,
        "null_mean": float(np.mean(macro_null[k])),
        "null_ci_lower": macro_ci_low,
        "null_ci_upper": macro_ci_high,
        "permutation_p_greater": macro_p,
    })

    permutation_rows.append({
        "metric": f"Micro Recall@{k}",
        "K": k,
        "observed": observed_micro,
        "null_mean": float(np.mean(micro_null[k])),
        "null_ci_lower": micro_ci_low,
        "null_ci_upper": micro_ci_high,
        "permutation_p_greater": micro_p,
    })

observed_mrr = float(mrr_summary.loc[0, "mrr_model"])
mrr_perm_p = (np.sum(mrr_null >= observed_mrr) + 1) / (n_permutations + 1)
mrr_ci_low, mrr_ci_high = np.percentile(mrr_null, [2.5, 97.5])

permutation_rows.append({
    "metric": "MRR",
    "K": np.nan,
    "observed": observed_mrr,
    "null_mean": float(np.mean(mrr_null)),
    "null_ci_lower": mrr_ci_low,
    "null_ci_upper": mrr_ci_high,
    "permutation_p_greater": mrr_perm_p,
})

permutation_summary = pd.DataFrame(permutation_rows)

recall_mask = permutation_summary["metric"].str.contains("Recall", na=False)
permutation_summary.loc[recall_mask, "permutation_q_bh"] = benjamini_hochberg(
    permutation_summary.loc[recall_mask, "permutation_p_greater"]
)
permutation_summary.loc[~recall_mask, "permutation_q_bh"] = permutation_summary.loc[
    ~recall_mask, "permutation_p_greater"
]
permutation_summary["star"] = permutation_summary["permutation_q_bh"].apply(p_to_star)

for k in ks:
    macro_row = permutation_summary[permutation_summary["metric"] == f"Macro Recall@{k}"].iloc[0]
    micro_row = permutation_summary[permutation_summary["metric"] == f"Micro Recall@{k}"].iloc[0]

    summary.loc[summary["K"] == k, "macro_random_null_ci_lower"] = macro_row["null_ci_lower"]
    summary.loc[summary["K"] == k, "macro_random_null_ci_upper"] = macro_row["null_ci_upper"]
    summary.loc[summary["K"] == k, "macro_permutation_p"] = macro_row["permutation_p_greater"]
    summary.loc[summary["K"] == k, "macro_permutation_q"] = macro_row["permutation_q_bh"]

    summary.loc[summary["K"] == k, "micro_random_null_ci_lower"] = micro_row["null_ci_lower"]
    summary.loc[summary["K"] == k, "micro_random_null_ci_upper"] = micro_row["null_ci_upper"]
    summary.loc[summary["K"] == k, "micro_permutation_p"] = micro_row["permutation_p_greater"]
    summary.loc[summary["K"] == k, "micro_permutation_q"] = micro_row["permutation_q_bh"]

mrr_summary["mrr_random_null_ci_lower"] = mrr_ci_low
mrr_summary["mrr_random_null_ci_upper"] = mrr_ci_high
mrr_summary["mrr_permutation_p"] = mrr_perm_p
mrr_summary["mrr_permutation_q"] = mrr_perm_p
mrr_summary["mrr_star"] = p_to_star(mrr_perm_p)

positive_rank_df = rank_df[rank_df["is_positive"]].copy()

meta_cols = [
    c for c in [
        "SITE_KEY",
        "INDEX",
        "GENE",
        "PROTEIN",
        "ACC_ID",
        "MOD_RSD",
        "RESIDUE",
        "POSITION",
        "LABEL",
        "Transport_Direction",
    ]
    if c in matched_pos.columns
]

positive_rank_df = positive_rank_df.merge(
    matched_pos[meta_cols],
    on="SITE_KEY",
    how="left",
    suffixes=("_pred", "_positive"),
)

for k in ks:
    positive_rank_df[f"hit_at_{k}"] = positive_rank_df["rank"] <= k

known_positive_sites = set(pos["SITE_KEY"])
if exclude_known_positive_from_cluster:
    cluster_for_eval = cluster[~cluster["SITE_KEY"].isin(known_positive_sites)].copy()
else:
    cluster_for_eval = cluster.copy()

cluster_for_eval = assign_cluster_ids(cluster_for_eval, gap=cluster_gap_for_auto_id)

cluster_for_eval["in_prediction_table"] = cluster_for_eval["SITE_KEY"].isin(pred_site_set)

matched_cluster = cluster_for_eval[cluster_for_eval["in_prediction_table"]].copy()
missing_cluster = cluster_for_eval[~cluster_for_eval["in_prediction_table"]].copy()

if matched_cluster.empty:
    raise ValueError(
        "No cluster supported sites matched the prediction table. "
        "Check SITE_KEY construction or set exclude_known_positive_from_cluster=False."
    )

pred_ranked = build_pred_ranked(pred, score_col)
per_protein_cluster, protein_cluster_summary, cluster_by_acc = compute_protein_level_cluster_summary(
    matched_cluster=matched_cluster,
    pred_ranked=pred_ranked,
    top_percentiles=top_percentiles,
)
protein_cluster_perm = run_protein_cluster_fold_permutation(
    pred_ranked=pred_ranked,
    cluster_by_acc=cluster_by_acc,
    top_percentiles=top_percentiles,
    n_permutations=n_permutations,
    random_seed=random_seed,
)
protein_cluster_summary = protein_cluster_summary.merge(
    protein_cluster_perm,
    on="top_percentile",
    how="left",
)

for pct in top_percentiles:
    pct_key = f"pct_{int(pct * 100):02d}"
    recall_col = f"recall_in_top_{pct_key}"
    random_col = f"random_recall_{pct_key}"
    wilcoxon_p = safe_wilcoxon_greater(
        per_protein_cluster[recall_col].astype(float),
        per_protein_cluster[random_col].astype(float),
    )
    protein_cluster_summary.loc[
        protein_cluster_summary["top_percentile"] == pct,
        "wilcoxon_p_greater",
    ] = wilcoxon_p

protein_cluster_summary["permutation_q_bh"] = benjamini_hochberg(
    protein_cluster_summary["permutation_p_greater"]
)
protein_cluster_summary["wilcoxon_q_bh"] = benjamini_hochberg(
    protein_cluster_summary["wilcoxon_p_greater"]
)
protein_cluster_summary["enrichment_star"] = protein_cluster_summary["permutation_p_greater"].apply(p_to_star)

cluster_rank_df = matched_cluster.merge(
    pred_ranked[
        [
            "SITE_KEY",
            "ACC_ID",
            score_col,
            "rank_within_protein",
            "candidate_n_within_protein",
            "global_rank",
            "global_percentile",
        ]
    ],
    on=["SITE_KEY", "ACC_ID"],
    how="inner",
    suffixes=("_cluster", "_pred"),
)
cluster_rank_df["rank_percentile_within_protein"] = (
    cluster_rank_df["rank_within_protein"] / cluster_rank_df["candidate_n_within_protein"]
)

cluster_meta_cols = [
    c for c in [
        "SITE_KEY",
        "CLUSTER_ID",
        "INDEX",
        "GENE",
        "PROTEIN",
        "ACC_ID",
        "MOD_RSD",
        "RESIDUE",
        "POSITION",
        "LABEL",
        "Transport_Direction",
        "PMID",
        "Literature_title",
    ]
    if c in matched_cluster.columns and c not in cluster_rank_df.columns
]

if cluster_meta_cols:
    cluster_rank_df = cluster_rank_df.merge(
        matched_cluster[cluster_meta_cols],
        on="SITE_KEY",
        how="left",
        suffixes=("", "_meta"),
    )

cluster_score_cols = [
    "SITE_KEY",
    "ACC_ID",
    score_col,
    "global_rank",
    "global_percentile",
    "rank_within_protein",
    "candidate_n_within_protein",
]

cluster_scored = matched_cluster.merge(
    pred_ranked[cluster_score_cols],
    on=["SITE_KEY", "ACC_ID"],
    how="inner",
    suffixes=("", "_pred"),
)

if cluster_scored.empty:
    raise ValueError("No cluster sites remained after merging with ranked prediction table.")

rng = np.random.default_rng(random_seed)

cluster_rows = []
skipped_cluster_rows = []
random_score_arrays = []
random_rank_arrays = []

all_matched_cluster_sites = set(cluster_scored["SITE_KEY"])

for cluster_id, cg in cluster_scored.groupby("CLUSTER_ID"):
    cg = cg.sort_values(score_col, ascending=False).copy()

    acc_ids = cg["ACC_ID"].dropna().astype(str).unique()
    if len(acc_ids) != 1:
        skipped_cluster_rows.append({
            "CLUSTER_ID": cluster_id,
            "reason": "multiple_acc_ids",
            "ACC_ID": ";".join(sorted(acc_ids)),
            "matched_site_n": len(cg),
        })
        continue

    acc_id = acc_ids[0]
    protein_df = pred_ranked[pred_ranked["ACC_ID"] == acc_id].copy()

    current_cluster_sites = set(cg["SITE_KEY"])
    cluster_size = len(current_cluster_sites)

    background_pool, background_mode = select_background_pool(
        protein_df=protein_df,
        current_cluster_sites=current_cluster_sites,
        known_positive_sites=known_positive_sites,
        all_cluster_sites=all_matched_cluster_sites,
        allow_relaxed=allow_relaxed_background,
    )

    if len(background_pool) < cluster_size:
        skipped_cluster_rows.append({
            "CLUSTER_ID": cluster_id,
            "reason": "insufficient_background",
            "ACC_ID": acc_id,
            "matched_site_n": len(cg),
            "cluster_size": cluster_size,
            "background_n": len(background_pool),
            "background_mode": background_mode,
        })
        continue

    best_idx = cg[score_col].idxmax()
    best_row = cg.loc[best_idx]

    best_rank_within_protein = int(cg["rank_within_protein"].min())
    candidate_n = int(cg["candidate_n_within_protein"].iloc[0])
    best_rank_percentile = best_rank_within_protein / candidate_n if candidate_n > 0 else np.nan

    cluster_score_max = float(cg[score_col].max())
    cluster_score_mean = float(cg[score_col].mean())
    cluster_score_top2_mean = float(cg[score_col].sort_values(ascending=False).head(2).mean())

    bg_scores = background_pool[score_col].astype(float).to_numpy()
    bg_ranks = background_pool["rank_within_protein"].astype(int).to_numpy()

    random_max_scores = np.zeros(n_permutations, dtype=float)
    random_best_ranks = np.zeros(n_permutations, dtype=float)

    for b in range(n_permutations):
        sampled_idx = rng.choice(len(background_pool), size=cluster_size, replace=False)
        random_max_scores[b] = float(np.max(bg_scores[sampled_idx]))
        random_best_ranks[b] = float(np.min(bg_ranks[sampled_idx]))

    empirical_p_score = empirical_greater_p(random_max_scores, cluster_score_max)
    empirical_p_rank = empirical_less_p(random_best_ranks, best_rank_within_protein)

    row = {
        "CLUSTER_ID": cluster_id,
        "ACC_ID": acc_id,
        "cluster_site_n_matched": cluster_size,
        "background_n": len(background_pool),
        "background_mode": background_mode,
        "cluster_sites": ";".join(cg["SITE_KEY"].astype(str).tolist()),
        "cluster_positions": ";".join(cg["POSITION"].astype(str).tolist()) if "POSITION" in cg.columns else "",
        "cluster_score_max": cluster_score_max,
        "cluster_score_mean": cluster_score_mean,
        "cluster_score_top2_mean": cluster_score_top2_mean,
        "best_site_key": str(best_row["SITE_KEY"]),
        "best_site_score": float(best_row[score_col]),
        "best_site_position": int(best_row["POSITION"]) if "POSITION" in cg.columns and pd.notna(best_row["POSITION"]) else np.nan,
        "protein_candidate_n": candidate_n,
        "best_rank_within_protein": best_rank_within_protein,
        "best_rank_percentile": best_rank_percentile,
        "random_max_score_mean": float(np.mean(random_max_scores)),
        "random_max_score_median": float(np.median(random_max_scores)),
        "random_max_score_ci_lower": float(np.percentile(random_max_scores, 2.5)),
        "random_max_score_ci_upper": float(np.percentile(random_max_scores, 97.5)),
        "random_best_rank_mean": float(np.mean(random_best_ranks)),
        "random_best_rank_median": float(np.median(random_best_ranks)),
        "random_best_rank_ci_lower": float(np.percentile(random_best_ranks, 2.5)),
        "random_best_rank_ci_upper": float(np.percentile(random_best_ranks, 97.5)),
        "empirical_p_score_greater": empirical_p_score,
        "empirical_p_rank_less": empirical_p_rank,
    }

    for col in ["GENE", "PROTEIN", "PMID", "Literature_title", "Transport_Direction", "LABEL"]:
        if col in cg.columns:
            values = cg[col].dropna().astype(str).unique()
            row[col] = ";".join(sorted(values))

    for k in ks:
        row[f"hit_at_{k}"] = best_rank_within_protein <= k

    cluster_rows.append(row)
    random_score_arrays.append(random_max_scores)
    random_rank_arrays.append(random_best_ranks)

cluster_level_summary = pd.DataFrame(cluster_rows)
skipped_cluster_df = pd.DataFrame(skipped_cluster_rows)

if cluster_level_summary.empty:
    raise ValueError(
        "No cluster was available for cluster level validation. "
        "Check CLUSTER_ID assignment and background pool size."
    )

cluster_level_summary["empirical_q_score_greater"] = benjamini_hochberg(
    cluster_level_summary["empirical_p_score_greater"]
)
cluster_level_summary["empirical_q_rank_less"] = benjamini_hochberg(
    cluster_level_summary["empirical_p_rank_less"]
)
cluster_level_summary["score_star"] = cluster_level_summary["empirical_q_score_greater"].apply(p_to_star)
cluster_level_summary["rank_star"] = cluster_level_summary["empirical_q_rank_less"].apply(p_to_star)

random_score_matrix = np.vstack(random_score_arrays)
random_rank_matrix = np.vstack(random_rank_arrays)

cluster_hit_rows = []

for k in ks:
    observed_hit_rate = float(cluster_level_summary[f"hit_at_{k}"].mean())
    random_hit_rates = (random_rank_matrix <= k).mean(axis=0)

    p_value = empirical_greater_p(random_hit_rates, observed_hit_rate)
    ci_lower, ci_upper = np.percentile(random_hit_rates, [2.5, 97.5])

    cluster_hit_rows.append({
        "K": k,
        "observed_hit_rate": observed_hit_rate,
        "observed_hit_n": int(cluster_level_summary[f"hit_at_{k}"].sum()),
        "cluster_n": len(cluster_level_summary),
        "random_hit_rate_mean": float(np.mean(random_hit_rates)),
        "random_hit_rate_median": float(np.median(random_hit_rates)),
        "random_hit_rate_ci_lower": float(ci_lower),
        "random_hit_rate_ci_upper": float(ci_upper),
        "empirical_p_greater": p_value,
    })

cluster_hit_summary = pd.DataFrame(cluster_hit_rows)
cluster_hit_summary["empirical_q_bh"] = benjamini_hochberg(cluster_hit_summary["empirical_p_greater"])
cluster_hit_summary["star"] = cluster_hit_summary["empirical_q_bh"].apply(p_to_star)

observed_mean_cluster_max_score = float(cluster_level_summary["cluster_score_max"].mean())
random_mean_cluster_max_scores = random_score_matrix.mean(axis=0)
score_global_p = empirical_greater_p(random_mean_cluster_max_scores, observed_mean_cluster_max_score)
score_global_ci_lower, score_global_ci_upper = np.percentile(random_mean_cluster_max_scores, [2.5, 97.5])

wilcoxon_p = safe_wilcoxon_greater(
    cluster_level_summary["cluster_score_max"],
    cluster_level_summary["random_max_score_mean"],
)

cluster_score_summary = pd.DataFrame([{
    "cluster_n": len(cluster_level_summary),
    "observed_mean_cluster_max_score": observed_mean_cluster_max_score,
    "random_mean_cluster_max_score": float(np.mean(random_mean_cluster_max_scores)),
    "random_median_cluster_max_score": float(np.median(random_mean_cluster_max_scores)),
    "random_ci_lower": float(score_global_ci_lower),
    "random_ci_upper": float(score_global_ci_upper),
    "permutation_p_greater": score_global_p,
    "wilcoxon_p_greater": wilcoxon_p,
    "permutation_star": p_to_star(score_global_p),
    "wilcoxon_star": p_to_star(wilcoxon_p),
}])

summary.to_csv(out_dir / "single_model_recall_summary.csv", index=False)
mrr_summary.to_csv(out_dir / "single_model_mrr_summary.csv", index=False)
per_protein.to_csv(out_dir / "single_model_per_protein_metrics.csv", index=False)
positive_rank_df.to_csv(out_dir / "single_model_positive_site_ranks.csv", index=False)
missing_pos.to_csv(out_dir / "single_model_missing_positive_sites.csv", index=False)
fisher_summary.to_csv(out_dir / "single_model_topk_fisher_enrichment.csv", index=False)
permutation_summary.to_csv(out_dir / "single_model_permutation_test_summary.csv", index=False)
protein_cluster_summary.to_csv(out_dir / "single_model_cluster_supported_protein_enrichment.csv", index=False)
per_protein_cluster.to_csv(out_dir / "single_model_cluster_supported_per_protein_metrics.csv", index=False)
cluster_rank_df.to_csv(out_dir / "single_model_cluster_supported_site_ranks.csv", index=False)
matched_cluster.to_csv(out_dir / "single_model_cluster_supported_matched_sites.csv", index=False)
missing_cluster.to_csv(out_dir / "single_model_cluster_supported_missing_sites.csv", index=False)

cluster_level_summary.to_csv(out_dir / "single_model_cluster_level_summary.csv", index=False)
cluster_hit_summary.to_csv(out_dir / "single_model_cluster_level_hit_at_k_summary.csv", index=False)
cluster_score_summary.to_csv(out_dir / "single_model_cluster_level_score_summary.csv", index=False)
skipped_cluster_df.to_csv(out_dir / "single_model_cluster_level_skipped_clusters.csv", index=False)

model_color = "#2F6C9F"
random_color = "#8A8A8A"
cluster_color = "#3F88C5"
line_color = "#B0B0B0"

fig, ax = plt.subplots(figsize=(4.8, 3.8))
ax.plot(
    summary["K"],
    summary["macro_recall_model"],
    marker="o",
    linewidth=2.2,
    markersize=5,
    color=model_color,
    label="Model",
)
ax.plot(
    summary["K"],
    summary["macro_recall_random"],
    marker="o",
    linewidth=2.0,
    markersize=5,
    color=random_color,
    label="Random",
)
ax.fill_between(
    summary["K"],
    summary["macro_random_null_ci_lower"],
    summary["macro_random_null_ci_upper"],
    color=random_color,
    alpha=0.18,
    linewidth=0,
)
for _, row in summary.iterrows():
    ax.text(
        row["K"],
        row["macro_recall_model"] + 0.035,
        p_to_star(row["macro_permutation_q"]),
        ha="center",
        va="bottom",
        fontsize=10 + FONT_SIZE_SHIFT,
        fontweight="regular",
    )
ax.set_xlabel("K")
ax.set_ylabel("Macro Recall@K")
ax.set_xticks(ks)
ax.set_ylim(0, min(1.08, max(summary["macro_recall_model"].max(), summary["macro_random_null_ci_upper"].max()) + 0.15))
ax.legend(frameon=False, loc="lower right")
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_macro_recall_at_k")

fig, ax = plt.subplots(figsize=(4.8, 3.8))
ax.plot(
    summary["K"],
    summary["micro_recall_model"],
    marker="o",
    linewidth=2.2,
    markersize=5,
    color=model_color,
    label="Model",
)
ax.plot(
    summary["K"],
    summary["micro_recall_random"],
    marker="o",
    linewidth=2.0,
    markersize=5,
    color=random_color,
    label="Random",
)
ax.fill_between(
    summary["K"],
    summary["micro_random_null_ci_lower"],
    summary["micro_random_null_ci_upper"],
    color=random_color,
    alpha=0.18,
    linewidth=0,
)
for _, row in summary.iterrows():
    ax.text(
        row["K"],
        row["micro_recall_model"] + 0.035,
        p_to_star(row["micro_permutation_q"]),
        ha="center",
        va="bottom",
        fontsize=10 + FONT_SIZE_SHIFT,
        fontweight="regular",
    )
ax.set_xlabel("K")
ax.set_ylabel("Micro Recall@K")
ax.set_xticks(ks)
ax.set_ylim(0, min(1.08, max(summary["micro_recall_model"].max(), summary["micro_random_null_ci_upper"].max()) + 0.15))
ax.legend(frameon=False, loc="lower right")
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_micro_recall_at_k")

fig, ax = plt.subplots(figsize=(2.7, 3.12))
bar_x = np.arange(2)
bar_values = [mrr_summary.loc[0, "mrr_random"], mrr_summary.loc[0, "mrr_model"]]
bar_colors = [random_color, model_color]

ax.bar(bar_x, bar_values, width=0.56, color=bar_colors, edgecolor="black", linewidth=0.7)

random_yerr_lower = mrr_summary.loc[0, "mrr_random"] - mrr_summary.loc[0, "mrr_random_null_ci_lower"]
random_yerr_upper = mrr_summary.loc[0, "mrr_random_null_ci_upper"] - mrr_summary.loc[0, "mrr_random"]
ax.errorbar(
    [0],
    [mrr_summary.loc[0, "mrr_random"]],
    yerr=[[random_yerr_lower], [random_yerr_upper]],
    fmt="none",
    color="black",
    capsize=3,
    linewidth=1.0,
)

mrr_star = mrr_summary.loc[0, "mrr_star"]
y_max_mrr = max(bar_values + [mrr_summary.loc[0, "mrr_random_null_ci_upper"]])
ax.plot([0, 0, 1, 1], [y_max_mrr + 0.035, y_max_mrr + 0.055, y_max_mrr + 0.055, y_max_mrr + 0.035], color="black", linewidth=1.0)
ax.text(0.5, y_max_mrr + 0.065, mrr_star, ha="center", va="bottom", fontsize=11 + FONT_SIZE_SHIFT, fontweight="regular")

ax.set_xticks(bar_x)
ax.set_xticklabels(["Random", "Model"])
ax.set_ylabel("MRR")
ax.set_xlim(-0.45, 1.45)
ax.set_ylim(0, min(1.08, y_max_mrr + 0.18))
clean_axes(ax)
fig.subplots_adjust(left=0.24, right=0.92, bottom=0.10, top=0.96)
fig.savefig(out_dir / "single_model_mrr.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_dir / "single_model_mrr.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_dir / "single_model_mrr.svg", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.8), sharex=True, sharey=False)
axes = axes.ravel()

x_grid = np.linspace(0, 1, 500)
recall_palette = ["#B5D0E6", "#7FB0D8", "#3F88C5", "#0B4F8A"]

for i, k in enumerate(ks):
    ax = axes[i]
    values = per_protein[f"recall_at_{k}"].astype(float).to_numpy()
    density = safe_kde(values, x_grid)

    ax.plot(x_grid, density, color=recall_palette[i], linewidth=2.2)
    ax.fill_between(x_grid, 0, density, color=recall_palette[i], alpha=0.22, linewidth=0)

    macro_value = float(summary.loc[summary["K"] == k, "macro_recall_model"].iloc[0])
    ax.axvline(macro_value, color=random_color, linestyle=":", linewidth=1.4)

    density_peak = float(np.max(density)) if len(density) else 0.0
    y_upper = max(density_peak * 1.18, 0.05)

    ax.set_title(f"Recall@{k} (n={len(values)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_upper)
    if i in [0, 2]:
        ax.set_ylabel("Density")
    if i in [2, 3]:
        ax.set_xlabel("Per protein Recall")
    clean_axes(ax)

fig.subplots_adjust(left=0.12, right=0.96, bottom=0.10, top=0.92, wspace=0.38, hspace=0.48)
save_figure(fig, out_dir / "single_model_per_protein_recall_density_panels")

x = np.arange(len(top_percentiles))
x_labels = [f"{int(p * 100)}%" for p in top_percentiles]

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot(
    x,
    protein_cluster_summary["fold_enrichment"],
    marker="o",
    linewidth=2.2,
    markersize=5,
    color=cluster_color,
)
ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)

for i, row in protein_cluster_summary.iterrows():
    ax.text(
        i,
        row["fold_enrichment"] * 1.04,
        row["enrichment_star"],
        ha="center",
        va="bottom",
        fontsize=10 + FONT_SIZE_SHIFT,
        fontweight="regular",
    )

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Top percentile")
ax.set_ylabel("Fold enrichment")
ax.set_ylim(0, protein_cluster_summary["fold_enrichment"].max() * 1.25)
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_cluster_supported_protein_enrichment")

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.plot(
    x,
    protein_cluster_summary["recall_in_top"],
    marker="o",
    linewidth=2.2,
    markersize=5,
    color=cluster_color,
    label="Model",
)
ax.plot(
    x,
    protein_cluster_summary["random_recall"],
    marker="o",
    linewidth=2.0,
    markersize=5,
    color=random_color,
    label="Random",
)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_xlabel("Top percentile")
ax.set_ylabel("Macro recall of cluster sites")
ax.set_ylim(0, min(1.05, max(protein_cluster_summary["recall_in_top"].max(), protein_cluster_summary["random_recall"].max()) * 1.25))
ax.legend(frameon=False, loc="upper left")
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_cluster_supported_protein_recovery")

fig, ax = plt.subplots(figsize=(4.8, 3.8))
ax.plot(
    cluster_hit_summary["K"],
    cluster_hit_summary["observed_hit_rate"],
    marker="o",
    linewidth=2.2,
    markersize=5,
    color=cluster_color,
    label="Model",
)
ax.plot(
    cluster_hit_summary["K"],
    cluster_hit_summary["random_hit_rate_mean"],
    marker="o",
    linewidth=2.0,
    markersize=5,
    color=random_color,
    label="Random",
)
ax.fill_between(
    cluster_hit_summary["K"],
    cluster_hit_summary["random_hit_rate_ci_lower"],
    cluster_hit_summary["random_hit_rate_ci_upper"],
    color=random_color,
    alpha=0.18,
    linewidth=0,
)
for _, row in cluster_hit_summary.iterrows():
    y_text = row["observed_hit_rate"] + 0.035
    ax.text(
        row["K"],
        y_text,
        row["star"],
        ha="center",
        va="bottom",
        fontsize=10 + FONT_SIZE_SHIFT,
        fontweight="regular",
    )

ax.set_xlabel("K")
ax.set_ylabel("Cluster Hit@K")
ax.set_xticks(ks)
y_upper = max(cluster_hit_summary["observed_hit_rate"].max(), cluster_hit_summary["random_hit_rate_ci_upper"].max()) + 0.15
ax.set_ylim(0, min(1.08, y_upper))
ax.legend(frameon=False, loc="lower right")
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_cluster_level_hit_at_k")

fig, ax = plt.subplots(figsize=(3.8, 3.8))
paired_df = cluster_level_summary.copy()

x_random = np.zeros(len(paired_df))
x_model = np.ones(len(paired_df))

for i, row in paired_df.iterrows():
    ax.plot(
        [0, 1],
        [row["random_max_score_mean"], row["cluster_score_max"]],
        color=line_color,
        alpha=0.65,
        linewidth=0.9,
        zorder=1,
    )

ax.scatter(
    x_random,
    paired_df["random_max_score_mean"],
    s=28,
    color=random_color,
    edgecolor="black",
    linewidth=0.4,
    zorder=2,
    label="Random",
)
ax.scatter(
    x_model,
    paired_df["cluster_score_max"],
    s=32,
    color=cluster_color,
    edgecolor="black",
    linewidth=0.4,
    zorder=3,
    label="Cluster",
)

y_max = max(paired_df["cluster_score_max"].max(), paired_df["random_max_score_mean"].max())
y_min = min(paired_df["cluster_score_max"].min(), paired_df["random_max_score_mean"].min())
y_margin = max((y_max - y_min) * 0.18, 0.08)

star = cluster_score_summary.loc[0, "permutation_star"]
ax.plot([0, 0, 1, 1], [y_max + y_margin * 0.35, y_max + y_margin * 0.55, y_max + y_margin * 0.55, y_max + y_margin * 0.35], color="black", linewidth=1.0)
ax.text(0.5, y_max + y_margin * 0.65, star, ha="center", va="bottom", fontsize=11 + FONT_SIZE_SHIFT, fontweight="regular")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Random", "Cluster"])
ax.set_ylabel("Max prediction score")
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(max(0, y_min - y_margin * 0.4), min(1.05, y_max + y_margin * 1.25))
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_cluster_level_max_score_paired")

rank_plot_df = cluster_level_summary.sort_values("best_rank_percentile", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.2, 3.8))
x_rank = np.arange(len(rank_plot_df))
ax.scatter(
    x_rank,
    rank_plot_df["best_rank_percentile"],
    s=34,
    color=cluster_color,
    edgecolor="black",
    linewidth=0.4,
    zorder=3,
)
ax.axhline(0.05, color=random_color, linestyle=":", linewidth=1.2)
ax.axhline(0.10, color=random_color, linestyle=":", linewidth=1.2)

ax.set_xlabel("Cluster")
ax.set_ylabel("Best rank percentile")
ax.set_xticks([])
ax.set_ylim(0, min(1.0, max(rank_plot_df["best_rank_percentile"].max() * 1.18, 0.12)))
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_cluster_level_best_rank_percentile")

forest_df = cluster_level_summary.copy()
forest_df["score_delta"] = forest_df["cluster_score_max"] - forest_df["random_max_score_mean"]
forest_df = forest_df.sort_values("score_delta", ascending=True).reset_index(drop=True)

if "GENE" in forest_df.columns:
    forest_df["plot_label"] = forest_df["GENE"].astype(str) + " | " + forest_df["CLUSTER_ID"].astype(str)
else:
    forest_df["plot_label"] = forest_df["CLUSTER_ID"].astype(str)

max_label_len = 36
forest_df["plot_label"] = forest_df["plot_label"].apply(lambda x: x[:max_label_len] + "..." if len(x) > max_label_len else x)

fig_height = max(4.2, 0.28 * len(forest_df) + 1.2)
fig, ax = plt.subplots(figsize=(6.2, fig_height))

y_pos = np.arange(len(forest_df))

ax.hlines(
    y=y_pos,
    xmin=forest_df["random_max_score_ci_lower"],
    xmax=forest_df["random_max_score_ci_upper"],
    color=random_color,
    linewidth=1.4,
    alpha=0.75,
    label="Random 95% CI",
)
ax.scatter(
    forest_df["random_max_score_mean"],
    y_pos,
    marker="|",
    s=80,
    color=random_color,
    linewidth=1.8,
    zorder=3,
    label="Random mean",
)
ax.scatter(
    forest_df["cluster_score_max"],
    y_pos,
    s=34,
    color=cluster_color,
    edgecolor="black",
    linewidth=0.4,
    zorder=4,
    label="Cluster max",
)

ax.set_yticks(y_pos)
ax.set_yticklabels(forest_df["plot_label"])
ax.set_xlabel("Max prediction score")
ax.set_xlim(0, 1.02)
ax.legend(frameon=False, loc="lower right")
clean_axes(ax)
plt.tight_layout()
save_figure(fig, out_dir / "single_model_cluster_level_score_forest")

print("Output directory:", out_dir)
print()
print("Recall summary")
print(summary)
print()
print("MRR summary")
print(mrr_summary)
print()
print("Top-K Fisher enrichment")
print(fisher_summary)
print()
print("Permutation summary")
print(permutation_summary)
print()
print("Protein-level cluster supported enrichment")
print(protein_cluster_summary)
print()
print("Cluster level hit summary")
print(cluster_hit_summary)
print()
print("Cluster level score summary")
print(cluster_score_summary)
print()
print("Cluster level summary")
print(cluster_level_summary)
print()
print("Skipped cluster summary")
print(skipped_cluster_df)