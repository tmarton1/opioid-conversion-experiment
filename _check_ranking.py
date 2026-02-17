import pandas as pd

pairs = pd.read_csv('results/ranking_vignettes_nielsen_units_pairs.csv')

# Focus on the three models of interest
targets = ['flash', 'qwen', 'mixtral']

print("=== ALL PAIRS AMONG flash, qwen, mixtral ===")
for _, row in pairs.iterrows():
    a = row['model_a'].split('/')[-1]
    b = row['model_b'].split('/')[-1]
    if any(t in a for t in targets) and any(t in b for t in targets):
        print(f"  {a:35s} vs {b:35s}")
        print(f"    tight_p={row['tight_p']:.6f}  broad_p={row['broad_p']:.6f}")
        print(f"    tight_bonf={row['tight_p_bonf']:.6f}  broad_bonf={row['broad_p_bonf']:.6f}")
        print(f"    a_sig_better={row['a_sig_better']}")
        print()

# Also show raw counts from the ranking CSV
print("=== RANKING DATA ===")
ranking = pd.read_csv('results/ranking_vignettes_nielsen_units.csv')
for _, row in ranking.iterrows():
    m = row['model'].split('/')[-1]
    if any(t in m for t in targets):
        print(f"  {m:35s}  tight={row['n_tight']}/{row['n_total']}={row['mean_tight_prop']:.3f}  "
              f"broad={row['n_broad']}/{row['n_total']}={row['mean_broad_prop']:.3f}  "
              f"raw_rank={row['raw_rank']}  adj_rank={row['adjusted_rank']}")
