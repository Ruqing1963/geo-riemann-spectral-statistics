#!/usr/bin/env python3
"""
Generate all figures for the paper in PDF format.

Usage:
    python generate_figures.py [--output-dir ../figures] [--format pdf]
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#333', 'ytick.color': '#333',
    'axes.edgecolor': '#666', 'grid.color': '#ddd',
    'font.family': 'serif', 'font.size': 11,
    'text.usetex': False,
})

C1, C2, C3, C4, C5 = '#2166ac', '#b2182b', '#4daf4a', '#ff7f00', '#984ea3'


def wigner_gue(s):
    return (32.0/np.pi**2)*s**2*np.exp(-4.0*s**2/np.pi)

def wigner_goe(s):
    return (np.pi/2.0)*s*np.exp(-np.pi*s**2/4.0)

def poisson_pdf(s):
    return np.exp(-s)

def make_cdf(pdf_func, s_max=6.0, n=10000):
    s = np.linspace(0, s_max, n)
    cdf = cumulative_trapezoid(pdf_func(s), s, initial=0)
    cdf /= cdf[-1]
    return interp1d(s, cdf, bounds_error=False, fill_value=(0,1))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format

    # Load data
    base = os.path.dirname(__file__)
    stages = pd.read_csv(os.path.join(base, '..', 'data', 'ics2023_stages.csv'))
    ages = np.sort(stages['base_age_ma'].values)[::-1]
    dt_raw = np.abs(np.diff(ages))
    dt = dt_raw[dt_raw > 0.1]
    dt_norm = dt / np.mean(dt)

    results_dir = os.path.join(base, '..', 'results')
    zeros_file = os.path.join(results_dir, 'riemann_zeros_500.csv')
    if os.path.exists(zeros_file):
        gamma = pd.read_csv(zeros_file)['gamma'].values
    else:
        print("  Run analysis.py first to generate Riemann zeros.")
        return
    raw_sp = np.diff(gamma)
    dens = (1/(2*np.pi)) * np.log(gamma[:-1]/(2*np.pi))
    dr = raw_sp * dens

    s_th = np.linspace(0, 4.0, 500)
    gue_cdf = make_cdf(wigner_gue)
    goe_cdf = make_cdf(wigner_goe)

    # ── Figure 1: Timeline ──
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = [C2 if d<3 else C4 if d<5 else C1 if d<10 else C3 for d in dt]
    ax.bar(range(len(dt)), dt, color=colors, alpha=0.75, edgecolor='#888', lw=0.3)
    ax.axhline(np.mean(dt), color='k', ls='--', lw=1, alpha=0.5,
               label=f'Mean = {np.mean(dt):.1f} Ma')
    ax.set_xlabel('Interval index (Cambrian base → Present)')
    ax.set_ylabel(r'$\Delta T$ (Ma)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'fig1_timeline.{ext}'), dpi=300)
    plt.close()

    # ── Figure 2: Histogram ──
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dt_norm, bins=np.linspace(0,3.5,20), density=True, alpha=0.6,
            color=C1, edgecolor=C1, label=f'Geological ($n$={len(dt_norm)})')
    ax.plot(s_th, wigner_gue(s_th), C2, lw=2.5, label=r'GUE ($\beta$=2)')
    ax.plot(s_th, wigner_goe(s_th), C4, lw=2, ls=':', label=r'GOE ($\beta$=1)')
    ax.plot(s_th, poisson_pdf(s_th), C3, lw=2.5, ls='--', label='Poisson')
    ax.set_xlabel(r'Normalized spacing $s$')
    ax.set_ylabel(r'$p(s)$')
    ax.set_xlim(0, 3.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'fig2_histogram.{ext}'), dpi=300)
    plt.close()

    # ── Figure 3: KDE ──
    fig, ax = plt.subplots(figsize=(7, 5))
    kde_g = gaussian_kde(dt_norm, bw_method=0.25)
    kde_r = gaussian_kde(dr, bw_method='silverman')
    s_k = np.linspace(0, 4.0, 500)
    ax.fill_between(s_k, kde_g(s_k), alpha=0.2, color=C1)
    ax.plot(s_k, kde_g(s_k), C1, lw=2.5, label='Geological KDE')
    ax.plot(s_k, kde_r(s_k), C5, lw=2, alpha=0.7, label='Riemann zeros KDE')
    ax.plot(s_th, wigner_gue(s_th), C2, lw=1.5, ls='--', label='GUE')
    ax.plot(s_th, wigner_goe(s_th), C4, lw=1.5, ls=':', label='GOE')
    ax.plot(s_th, poisson_pdf(s_th), C3, lw=1.5, ls='-.', label='Poisson')
    ax.set_xlabel(r'Normalized spacing $s$')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 3.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'fig3_kde.{ext}'), dpi=300)
    plt.close()

    # ── Figure 4: CDF ──
    fig, ax = plt.subplots(figsize=(7, 5))
    gs = np.sort(dt_norm)
    ge = np.arange(1, len(gs)+1)/len(gs)
    rs = np.sort(dr)
    re = np.arange(1, len(rs)+1)/len(rs)
    sc = np.linspace(0, 4.0, 2000)
    ax.step(gs, ge, C1, lw=2.5, label='Geological ECDF', where='post')
    ax.step(rs, re, C5, lw=1, label='Riemann ECDF', where='post', alpha=0.5)
    ax.plot(sc, gue_cdf(sc), C2, lw=2, ls='--', label='GUE CDF')
    ax.plot(sc, goe_cdf(sc), C4, lw=1.5, ls=':', label='GOE CDF')
    ax.plot(sc, 1-np.exp(-sc), C3, lw=2, ls='-.', label='Poisson CDF')
    ax.set_xlabel(r'$s$'); ax.set_ylabel(r'$F(s)$')
    ax.set_xlim(0, 3.5)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'fig4_cdf.{ext}'), dpi=300)
    plt.close()

    # ── Figure 5: P-P plot ──
    fig, ax = plt.subplots(figsize=(6, 6))
    emp = np.arange(1, len(gs)+1)/(len(gs)+1)
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.3)
    ax.scatter(goe_cdf(gs), emp, c=C4, s=30, alpha=0.8, zorder=4, label='vs GOE')
    ax.scatter(gue_cdf(gs), emp, c=C2, s=30, alpha=0.8, marker='s', zorder=3, label='vs GUE')
    ax.scatter(1-np.exp(-gs), emp, c=C3, s=30, alpha=0.6, marker='^', zorder=2, label='vs Poisson')
    ax.set_xlabel('Theoretical CDF'); ax.set_ylabel('Empirical CDF')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'fig5_pp.{ext}'), dpi=300)
    plt.close()

    # ── Figure 6: Bootstrap ──
    boot_file = os.path.join(results_dir, 'bootstrap_samples.npz')
    if os.path.exists(boot_file):
        d = np.load(boot_file)
        boot_r, boot_var = d['boot_r'], d['boot_var']
    else:
        rng = np.random.default_rng(42)
        def r_s(x): return np.mean(np.minimum(x[:-1],x[1:])/np.maximum(x[:-1],x[1:]))
        boot_r = np.array([r_s(rng.choice(dt_norm, len(dt_norm), True)) for _ in range(10000)])
        boot_var = np.array([np.var(rng.choice(dt_norm, len(dt_norm), True)) for _ in range(10000)])

    r_obs = np.mean(np.minimum(dt_norm[:-1],dt_norm[1:])/np.maximum(dt_norm[:-1],dt_norm[1:]))
    var_obs = np.var(dt_norm)

    fig, (a1,a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.hist(boot_r, 50, density=True, alpha=0.6, color=C1, edgecolor=C1)
    a1.axvline(0.386, color=C3, lw=2.5, ls='--', label='Poisson (0.386)')
    a1.axvline(0.536, color=C4, lw=2, ls=':', label='GOE (0.536)')
    a1.axvline(0.603, color=C2, lw=2.5, label='GUE (0.603)')
    a1.axvline(r_obs, color='k', lw=2, label=f'Observed ({r_obs:.3f})')
    ci_r = np.percentile(boot_r, [2.5, 97.5])
    a1.axvspan(ci_r[0], ci_r[1], alpha=0.1, color='k', label='95% CI')
    a1.set_xlabel(r'$\langle r \rangle$'); a1.set_ylabel('Density')
    a1.set_title(r'(a) Bootstrap $\langle r \rangle$'); a1.legend(fontsize=8); a1.grid(True, alpha=0.3)

    a2.hist(boot_var, 50, density=True, alpha=0.6, color=C1, edgecolor=C1)
    a2.axvline(1.0, color=C3, lw=2.5, ls='--', label='Poisson (1.0)')
    a2.axvline(0.178, color=C2, lw=2.5, label='GUE (~0.178)')
    a2.axvline(var_obs, color='k', lw=2, label=f'Observed ({var_obs:.3f})')
    ci_v = np.percentile(boot_var, [2.5, 97.5])
    a2.axvspan(ci_v[0], ci_v[1], alpha=0.1, color='k', label='95% CI')
    a2.set_xlabel(r'Var$(s)$'); a2.set_ylabel('Density')
    a2.set_title('(b) Bootstrap variance'); a2.legend(fontsize=8); a2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, f'fig6_bootstrap.{ext}'), dpi=300)
    plt.close()

    print(f"  All 6 figures saved to {os.path.abspath(args.output_dir)}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='../figures')
    parser.add_argument('--format', default='pdf', choices=['pdf','png','svg'])
    main(parser.parse_args())
