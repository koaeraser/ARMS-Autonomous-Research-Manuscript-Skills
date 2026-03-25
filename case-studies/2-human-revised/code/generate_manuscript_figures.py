"""
Generate the 4 missing manuscript figures from existing results CSVs.

Figures:
  1. fig_per_arm_comparison.pdf  — per-arm SE scatter, KG-CAR vs rMAP
  2. fig_sensitivity.pdf         — 3-panel sensitivity (rho, w_rob, KG weights)
  3. fig_ablation.pdf            — component contribution bar chart
  4. fig_design_oc_structured_v2.pdf — design OC curves (T1E & power vs true pC)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})


def fig_per_arm_comparison():
    """
    Per-arm squared error scatter: KG-CAR vs rMAP.
    Color = true MRD negativity rate.
    """
    df = pd.read_csv(RESULTS_DIR / "loocv_results.csv")

    kgcar = df[df['method'] == 'KG-CAR'][['trial_id', 'se', 'p_true']].set_index('trial_id')
    rmap = df[df['method'] == 'rMAP'][['trial_id', 'se']].set_index('trial_id')
    merged = kgcar.join(rmap, lsuffix='_kgcar', rsuffix='_rmap')

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    sc = ax.scatter(merged['se_rmap'], merged['se_kgcar'],
                    c=merged['p_true'], cmap='RdYlBu_r', s=50,
                    edgecolors='black', linewidth=0.4, zorder=3,
                    vmin=0.0, vmax=0.85)

    # Diagonal
    max_se = max(merged['se_rmap'].max(), merged['se_kgcar'].max()) * 1.05
    ax.plot([0, max_se], [0, max_se], 'k--', alpha=0.4, linewidth=0.8, zorder=1)

    ax.set_xlabel('rMAP Squared Error')
    ax.set_ylabel('KG-CAR Squared Error')
    ax.set_title('Per-Arm Squared Error: KG-CAR vs. rMAP')
    ax.set_xlim(-0.002, max_se)
    ax.set_ylim(-0.002, max_se)
    ax.set_aspect('equal')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('True MRD Negativity Rate')

    n_better = (merged['se_kgcar'] < merged['se_rmap']).sum()
    n_total = len(merged)
    ax.text(0.95, 0.05,
            f'KG-CAR wins: {n_better}/{n_total} arms',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white', alpha=0.8))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_per_arm_comparison.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  {out.name}")


def fig_sensitivity():
    """
    3-panel sensitivity: (a) rho, (b) w_rob, (c) KG weights.
    """
    df = pd.read_csv(RESULTS_DIR / "stress_test_sensitivity.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Sensitivity to rho
    ax = axes[0]
    rho_data = df[df['analysis'] == 'rho']
    rho_summary = rho_data.groupby('param_value').agg(
        MSE=('se', 'mean'),
        MSE_se=('se', lambda x: x.std() / np.sqrt(len(x))),
        Coverage=('covered', 'mean'),
    ).reset_index()
    rho_summary['param_value'] = rho_summary['param_value'].astype(float)
    rho_summary = rho_summary.sort_values('param_value')

    ax.errorbar(rho_summary['param_value'], rho_summary['MSE'],
                yerr=1.96 * rho_summary['MSE_se'],
                fmt='o-', color='#D95319', capsize=3, markersize=6)
    ax.set_xlabel(r'$\rho$ (fixed)')
    ax.set_ylabel('MSE')
    ax.set_title(r'(a) Sensitivity to $\rho$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Sensitivity to w_rob
    ax = axes[1]
    wrob_data = df[df['analysis'] == 'w_rob']
    wrob_summary = wrob_data.groupby('param_value').agg(
        MSE=('se', 'mean'),
        MSE_se=('se', lambda x: x.std() / np.sqrt(len(x))),
        Coverage=('covered', 'mean'),
    ).reset_index()
    wrob_summary['param_value'] = wrob_summary['param_value'].astype(float)
    wrob_summary = wrob_summary.sort_values('param_value')

    ax.errorbar(wrob_summary['param_value'], wrob_summary['MSE'],
                yerr=1.96 * wrob_summary['MSE_se'],
                fmt='s-', color='#D95319', capsize=3, markersize=6)
    ax.set_xlabel(r'$w_{\mathrm{rob}}$')
    ax.set_ylabel('MSE')
    ax.set_title(r'(b) Sensitivity to $w_{\mathrm{rob}}$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) KG dimension weights
    ax = axes[2]
    kg_data = df[df['analysis'] == 'kg_weights']
    kg_summary = kg_data.groupby('param_value').agg(
        MSE=('se', 'mean'),
    ).reset_index()
    kg_summary = kg_summary.sort_values('MSE', ascending=True)

    bars = ax.barh(range(len(kg_summary)), kg_summary['MSE'],
                   color='#D95319', edgecolor='black', linewidth=0.4,
                   height=0.6)
    ax.set_yticks(range(len(kg_summary)))
    ax.set_yticklabels(kg_summary['param_value'], fontsize=8)
    ax.set_xlabel('MSE')
    ax.set_title('(c) KG Dimension Weights')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_sensitivity.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  {out.name}")


def fig_ablation():
    """
    Component contribution analysis: MSE and CRPS for KG-CAR variants.
    """
    df = pd.read_csv(RESULTS_DIR / "stress_test_ablation.csv")

    summary = df.groupby('method').agg(
        MSE=('se', 'mean'),
        CRPS=('crps', 'mean'),
    ).reset_index()

    # Order: full first, then variants
    order = ['KG-CAR (full)', 'KG-CAR (no rob)', 'KG-CAR (no BYM)', 'BHM (no KG)']
    summary['method'] = pd.Categorical(summary['method'], categories=order, ordered=True)
    summary = summary.sort_values('method')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(summary))
    colors = ['#D95319', '#0072BD', '#77AC30', '#999999']
    labels = ['KG-CAR\n(full)', 'KG-CAR\n(no rob)', 'KG-CAR\n(no BYM)', 'BHM\n(no KG)']

    # MSE panel
    ax = axes[0]
    bars = ax.bar(x, summary['MSE'], color=colors, edgecolor='black',
                  linewidth=0.5, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MSE')
    ax.set_title('(a) Mean Squared Error')
    # Add value labels
    for bar, val in zip(bars, summary['MSE']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # CRPS panel
    ax = axes[1]
    bars = ax.bar(x, summary['CRPS'], color=colors, edgecolor='black',
                  linewidth=0.5, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('CRPS')
    ax.set_title('(b) Continuous Ranked Probability Score')
    for bar, val in zip(bars, summary['CRPS']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_ablation.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  {out.name}")


def fig_design_oc():
    """
    Design OC: 3x2 grid using permutation-based simulation.
    For each replicate: shuffle similarities -> KG-CAR center -> true pC = center.
    KG-CAR is centered at the shuffled center (near truth).
    rMAP is centered at the grand mean (fixed).
    Results binned by realized true pC, with fixed threshold gamma=0.90.
    """
    import pickle
    from scipy.special import expit, logit as sp_logit

    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from design_oc_structured_v2 import (
        compute_posterior_prob, compute_prob_matrix,
        shuffle_center, compute_unshuffled_noise,
    )

    # Load data
    DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data"
    trials_df = pd.read_csv(DATA_DIR / "trials_data.csv")
    with open(DATA_DIR / "similarity_matrices.pkl", 'rb') as f:
        sim_data = pickle.load(f)

    S_composite = sim_data['S_composite']
    trial_ids = sim_data['trial_ids']
    trials_indexed = trials_df.set_index('trial_id')
    p_obs = np.array([trials_indexed.loc[tid, 'p_hat'] for tid in trial_ids])
    theta_obs = sp_logit(p_obs)
    H = len(theta_obs)
    mu_rmap = np.mean(theta_obs)

    power_k = 1
    top_m = 5
    sigma2_kg = 1.00
    sigma2_rm = 1.06
    n_T = 50
    threshold = 0.90
    w_rob = 0.10
    vague_sd = 2.0

    # Compute KG centers to find reference arms
    sigma2_noise, kg_centers = compute_unshuffled_noise(
        S_composite, theta_obs, power_k, top_m)
    kg_centers_p = expit(kg_centers)

    # Find arms closest to target centers
    target_centers = [0.25, 0.51, 0.76]
    ref_arms = []
    for tc in target_centers:
        idx = np.argmin(np.abs(kg_centers_p - tc))
        ref_arms.append(idx)
        print(f"    Ref arm for p={tc}: idx={idx} "
              f"({trial_ids[idx]}, KG center={kg_centers_p[idx]:.3f})")

    ref_labels = [
        r'Low KG center ($\bar{p} = 0.25$)',
        r'Mid KG center ($\bar{p} = 0.51$)',
        r'High KG center ($\bar{p} = 0.76$)',
    ]
    nC_values = [10, 25]
    R = 10000

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))

    for row, (arm_idx, label) in enumerate(zip(ref_arms, ref_labels)):
        sims = S_composite[arm_idx, :].copy()

        for col, nC in enumerate(nC_values):
            ax = axes[row, col]
            print(f"    Simulating arm={arm_idx}, nC={nC}, R={R}...",
                  end=" ", flush=True)

            # Precompute rMAP P matrix (fixed center)
            P_rm = compute_prob_matrix(n_T, nC, mu_rmap, sigma2_rm,
                                       w_rob, vague_sd)

            rng = np.random.default_rng(42)
            records = []  # (true_pC, delta, kg_prob, rm_prob)

            for r in range(R):
                # Shuffle similarities -> KG-CAR center
                mu_kg = shuffle_center(theta_obs, sims, power_k, top_m,
                                       arm_idx, rng)

                # True pC = KG center exactly (KG correctly specified)
                true_pC = expit(mu_kg)

                for delta in [0.00, 0.15]:
                    true_pT = np.clip(true_pC + delta, 0.0, 0.999)
                    y_T = rng.binomial(n_T, true_pT)
                    y_C = rng.binomial(nC, true_pC)

                    # KG-CAR prob (centered at shuffled center)
                    prob_kg = compute_posterior_prob(
                        y_T, n_T, y_C, nC, mu_kg, sigma2_kg,
                        w_rob, vague_sd)

                    # rMAP prob (use precomputed matrix)
                    prob_rm = float(P_rm[y_T, y_C])

                    records.append((true_pC, delta, prob_kg, prob_rm))

            df_rec = pd.DataFrame(records,
                                  columns=['true_pC', 'delta', 'prob_kg', 'prob_rm'])

            # Bin by true_pC (0.10-width bins)
            df_rec['pC_bin'] = (df_rec['true_pC'] * 10).round() / 10  # 0.10 bins

            for delta, linestyle in [(0.0, '-'), (0.15, '--')]:
                sub = df_rec[df_rec['delta'] == delta]
                binned = sub.groupby('pC_bin').agg(
                    kg_rate=('prob_kg', lambda x: np.mean(x > threshold)),
                    rm_rate=('prob_rm', lambda x: np.mean(x > threshold)),
                    count=('prob_kg', 'count'),
                ).reset_index()
                # Filter bins with enough data
                binned = binned[binned['count'] >= 20].sort_values('pC_bin')

                lbl_suffix = 'T1E' if delta == 0 else r'Power ($\Delta=0.15$)'
                ax.plot(binned['pC_bin'], binned['kg_rate'],
                        f'o{linestyle}', color='#D95319', markersize=4,
                        linewidth=1.5,
                        label=f'KG-CAR {lbl_suffix}')
                ax.plot(binned['pC_bin'], binned['rm_rate'],
                        f's{linestyle}', color='#0072BD', markersize=4,
                        linewidth=1.5,
                        label=f'rMAP {lbl_suffix}')

            # Reference lines
            ax.axhline(y=0.10, color='gray', linestyle=':', alpha=0.5,
                       linewidth=0.8)
            ax.axvline(x=expit(mu_rmap), color='gray', linestyle='--',
                       alpha=0.3, linewidth=0.8, label='grand mean')

            ax.set_xlabel(r'True $p_C$ (binned)')
            ax.set_ylabel('Rate')
            ax.set_title(f'{label}, $n_C = {nC}$', fontsize=10)
            ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
            ax.set_xlim(0.15, 0.75)
            ax.set_ylim(-0.02, 0.78)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            print("done")

    plt.tight_layout()
    out = FIGURES_DIR / "fig_design_oc_structured_v2.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  {out.name}")


if __name__ == "__main__":
    print("Generating manuscript figures...")
    fig_per_arm_comparison()
    fig_sensitivity()
    fig_ablation()
    fig_design_oc()
    print(f"Done. Figures saved to: {FIGURES_DIR}")
