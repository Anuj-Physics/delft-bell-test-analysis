
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import os

data_files = {
    '20151201od_relaxed': ('20151201od.txt', 'old', True),
    'old_detector': ('bell_open_data_2_old_detector.txt', 'old', False),
    'new_detector': ('bell_open_data_2_new_detector.txt', 'new', False)
}


def check_entangled_state_distribution(filepath):
    data = np.loadtxt(filepath, delimiter=',', dtype=str)
    cols = data[:, 1:].astype(np.int64)
    click1_ch = cols[:, 3]
    click2_ch = cols[:, 5]
    psi_plus = np.sum(click1_ch == click2_ch)
    psi_min = np.sum(click1_ch != click2_ch)
    total = psi_plus + psi_min
    print(f"\nFile: {filepath}")
    print(f"ψ⁺ trials: {psi_plus} ({psi_plus/total:.2%})")
    print(f"ψ⁻ trials: {psi_min} ({psi_min/total:.2%})")

def load_and_filter(filepath, detector_label, relaxed=False):
    data = np.loadtxt(filepath, delimiter=',', dtype=str)
    cols = data[:, 1:].astype(np.int64)
    click1_time, click1_ch = cols[:, 2], cols[:, 3]
    click2_time, click2_ch = cols[:, 4], cols[:, 5]
    rnd_A, rnd_B = cols[:, 6], cols[:, 7]
    ro_time_A, ro_time_B = cols[:, 10], cols[:, 11]
    click_exc_A, click_exc_B = cols[:, 12], cols[:, 13]
    inv_A, inv_B = cols[:, 14], cols[:, 15]

    
    start_ch0 = 5426000 if detector_label == 'old' else 5425300
    start_ch1 = 5425100
    win_len = 50000
    sep = 250000
    psi_plus_len_ch0 = 4000
    psi_plus_len_ch1 = 2500
    ro_start = 10620
    ro_len = 3700
    max_invalid = 250

    psi_min = click1_ch != click2_ch

    if relaxed:
        valid = psi_min  
    else:
        win1 = (((start_ch0 <= click1_time) & (click1_time < start_ch0 + win_len) & (click1_ch == 0)) |
                ((start_ch1 <= click1_time) & (click1_time < start_ch1 + win_len) & (click1_ch == 1)))

        win2_min = (((start_ch0 + sep <= click2_time) & (click2_time < start_ch0 + sep + win_len) & (click2_ch == 0)) |
                    ((start_ch1 + sep <= click2_time) & (click2_time < start_ch1 + sep + win_len) & (click2_ch == 1)))

        event_ready = win1 & win2_min & psi_min
        valid = event_ready & ((inv_A == 0) | (inv_A > max_invalid)) & ((inv_B == 0) | (inv_B > max_invalid))
        valid &= (click_exc_A == 0) & (click_exc_B == 0)

    detect_A = (ro_time_A > ro_start) & (ro_time_A <= ro_start + ro_len)
    detect_B = (ro_time_B > ro_start) & (ro_time_B <= ro_start + ro_len)

    mask = valid
    a_i = rnd_A[mask]
    b_i = rnd_B[mask]
    x_i = (detect_A[mask] * 2 - 1)
    y_i = (detect_B[mask] * 2 - 1)
    return a_i, b_i, x_i, y_i


def analyze_bell_data(a_i, b_i, x_i, y_i, label):
    a_i, b_i, x_i, y_i = map(np.array, (a_i, b_i, x_i, y_i))
    n = len(a_i)
    c_i = ((-1)**(a_i*(b_i+1)) * (x_i*y_i) + 1)/2
    k = np.sum(c_i)
    tau = 5.4e-6*2
    ksi = 0.75 + 3*(tau + tau**2)
    p_val = 1 - binom.cdf(k-1, n, ksi)

    print(f"\n--- {label} ---")
    print(f"Valid ψ⁻ Trials (n): {n}, Bell wins (k): {k}, p-value: {p_val:.4f}")

    
    E, E_err = [], []
    for a, b in [(0,0),(0,1),(1,0),(1,1)]:
        mask = (a_i == a) & (b_i == b)
        if np.sum(mask) == 0:
            E.append(0); E_err.append(0)
        else:
            corr = x_i[mask] * y_i[mask]
            val = np.mean(corr)
            err = np.sqrt((1 - val**2) / len(corr))
            E.append(val); E_err.append(err)

    S = E[0] + E[1] + E[2] - E[3]
    S_err = np.sqrt(sum(np.square(E_err)))
    Z = (S - 2) / S_err if S_err > 0 else float('inf')

    print(f"CHSH S = {S:.3f} ± {S_err:.3f} → Z = {Z:.2f}σ")
    return n, S, S_err


all_a, all_b, all_x, all_y = [], [], [], []
n_vals, S_vals, S_errs = [], [], []

for key, (file, detector, relaxed) in data_files.items():
    if not os.path.exists(file):
        print(f"Missing file: {file}")
        continue
    check_entangled_state_distribution(file)
    a, b, x, y = load_and_filter(file, detector, relaxed=relaxed)
    all_a.append(a); all_b.append(b); all_x.append(x); all_y.append(y)
    n, S, S_err = analyze_bell_data(a, b, x, y, label=key)
    n_vals.append(n); S_vals.append(S); S_errs.append(S_err)


a_all = np.concatenate(all_a)
b_all = np.concatenate(all_b)
x_all = np.concatenate(all_x)
y_all = np.concatenate(all_y)
print("\n=== Combined ψ⁻ Dataset ===")
analyze_bell_data(a_all, b_all, x_all, y_all, label="Combined ψ⁻")

cum_n = np.cumsum(n_vals)
cum_S, cum_E = [], []
a_c, b_c, x_c, y_c = np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=int), np.array([],dtype=int)
for a, b, x, y in zip(all_a, all_b, all_x, all_y):
    a_c = np.concatenate([a_c, a])
    b_c = np.concatenate([b_c, b])
    x_c = np.concatenate([x_c, x])
    y_c = np.concatenate([y_c, y])
    _, S_now, err_now = analyze_bell_data(a_c, b_c, x_c, y_c, label="Running")
    cum_S.append(S_now); cum_E.append(err_now)

plt.figure(figsize=(7,4))
plt.errorbar(cum_n, cum_S, yerr=cum_E, fmt='-o', capsize=5)
plt.axhline(2, linestyle='--', color='r', label='Classical Limit')
plt.axhline(2*np.sqrt(2), linestyle='--', color='g', label='Tsirelson Bound')
plt.xscale('log')
plt.xlabel("Number of ψ⁻ Trials")
plt.ylabel("CHSH S")
plt.title("CHSH S vs Sample Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n Finished full CHSH ψ⁻ analysis with running plot and subset merging.")
