import seaborn as sns

def bootstrap_chsh_S(a, b, x, y, N_samples=1000):
    n = len(a)
    S_values = []
    for _ in range(N_samples):
        idx = np.random.choice(n, n, replace=True)
        a_s, b_s, x_s, y_s = a[idx], b[idx], x[idx], y[idx]
        
        E = []
        for a_, b_ in [(0,0), (0,1), (1,0), (1,1)]:
            mask = (a_s == a_) & (b_s == b_)
            if np.sum(mask) == 0:
                E.append(0)
            else:
                val = np.mean(x_s[mask] * y_s[mask])
                E.append(val)
        S = E[0] + E[1] + E[2] - E[3]
        S_values.append(S)
    
    return np.array(S_values)

boot_S = bootstrap_chsh_S(a_all, b_all, x_all, y_all, N_samples=5000)
conf_95 = np.percentile(boot_S, [2.5, 97.5])
mean_S = np.mean(boot_S)

plt.figure(figsize=(8,4))
sns.histplot(boot_S, kde=True, bins=40, color="skyblue", edgecolor="k")
plt.axvline(2, color='r', linestyle='--', label="Classical Bound (2)")
plt.axvline(mean_S, color='b', linestyle='-', label=f"Mean S = {mean_S:.3f}")
plt.axvline(conf_95[0], color='g', linestyle='--', label=f"95% CI: {conf_95[0]:.3f}")
plt.axvline(conf_95[1], color='g', linestyle='--', label=f"95% CI: {conf_95[1]:.3f}")
plt.title("Bootstrap Distribution of CHSH S (ψ⁻ Combined Data)")
plt.xlabel("CHSH S Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
