#!/usr/bin/env python3
import numpy as np, os
ad = os.path.join(os.path.dirname(__file__), 'results/K2103/analysis')
rn = np.load(f'{ad}/gnm_wt/resnums.npy')
si = int(np.argmin(np.abs(rn - 2103)))
for tag in ['gnm', 'anm']:
    is_anm = tag == 'anm'
    for state in ['wt', 'mut']:
        ev = np.load(f'{ad}/{tag}_{state}/eigenvalues.npy')
        ec = np.load(f'{ad}/{tag}_{state}/eigenvectors.npy')
        n_m = len(ev)
        if is_anm:
            n_r = ec.shape[0] // 3
            sf_site = []
            for i in range(n_m):
                v = ec[:, i].reshape(n_r, 3)
                sf_site.append((v[si]**2).sum() / ev[i])
        else:
            sf_site = [float(ec[si, i]**2 / ev[i]) for i in range(n_m)]
        total = sum(sf_site)
        pcts = [100*x/total for x in sf_site]
        print(f'{tag}_{state} sf_per_mode: {[round(x,4) for x in sf_site[:10]]}')
        print(f'  pct_at_site: {[round(x,1) for x in pcts[:10]]}')

gnm_ov = np.load(f'{ad}/comparison/gnm_mode_overlaps.npy')
anm_ov = np.load(f'{ad}/comparison/anm_mode_overlaps.npy')
print(f'GNM overlaps: {[round(float(x),4) for x in gnm_ov]}')
print(f'ANM overlaps: {[round(float(x),4) for x in anm_ov]}')

# Delta per-mode sqfluct at site
for tag in ['gnm', 'anm']:
    is_anm = tag == 'anm'
    ev_w = np.load(f'{ad}/{tag}_wt/eigenvalues.npy')
    ec_w = np.load(f'{ad}/{tag}_wt/eigenvectors.npy')
    ev_m = np.load(f'{ad}/{tag}_mut/eigenvalues.npy')
    ec_m = np.load(f'{ad}/{tag}_mut/eigenvectors.npy')
    n_m = len(ev_w)
    if is_anm:
        n_r = ec_w.shape[0] // 3
        wt_s = [float((ec_w[:, i].reshape(n_r,3)[si]**2).sum()/ev_w[i]) for i in range(n_m)]
        mt_s = [float((ec_m[:, i].reshape(n_r,3)[si]**2).sum()/ev_m[i]) for i in range(n_m)]
    else:
        wt_s = [float(ec_w[si,i]**2/ev_w[i]) for i in range(n_m)]
        mt_s = [float(ec_m[si,i]**2/ev_m[i]) for i in range(n_m)]
    delta = [m - w for w, m in zip(wt_s, mt_s)]
    print(f'{tag} delta_sf_per_mode_at_site: {[round(x,4) for x in delta[:10]]}')
