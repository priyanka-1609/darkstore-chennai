"""
Profit vs store limit K for 10,000 customers and 50 candidate sites.
Run once:  python k_sweep.py      -> writes k_sweep.csv (shown in the app)
"""
import pandas as pd
from data import generate_customers, make_candidates
from model import solve_grouped

REV, CPK = 90, 8
K_VALUES = [2, 5, 10, 20, 25]
DMAX_VALUES = [2, 3, 4, 8]
GAP_TARGET = {2: 0.0, 3: 0.0, 4: 0.0, 8: 0.0}   # % ; 0 = prove optimal

if __name__ == "__main__":
    cust = generate_customers(10000, seed=42)
    sites = make_candidates(cust, 50, seed=42)
    rows = []
    for dmax in DMAX_VALUES:
        for K in K_VALUES:
            kpi, s, c, r = solve_grouped(cust, sites, REV, CPK, dmax, K, time_limit_s=300, gap_pct=GAP_TARGET[dmax])
            rows.append(dict(max_dist_km=dmax, K=K, profit_per_day=round(r["profit"]),
                             stores_opened=r["n_open"], demand_served_pct=round(r["served_pct"], 1),
                             status=r["status"], gap_pct=round(r["gap_pct"], 3),
                             solve_s=round(r["build_s"] + r["solve_s"] + r["group_s"], 1)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("k_sweep.csv", index=False)
    print("Saved k_sweep.csv")
