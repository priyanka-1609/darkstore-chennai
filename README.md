# Dark Store Location Optimisation — Chennai

Streamlit app that decides **which candidate dark stores to open** and **which store serves each customer**
to maximise daily profit, with a **maximum store–customer distance** constraint (Assignment 1 enhancement).
Solved as a MILP with Google OR-Tools + SCIP.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app: inputs, KPI table, solution map, profit-vs-K chart, model & assumptions |
| `model.py` | MILP model (OR-Tools / SCIP) and result summary |
| `data.py` | Chennai customer generation + K-Means candidate sites + distance matrix |
| `k_sweep.py` | Runs K = 2, 5, 10, 20, 25 for 10,000 customers × 50 candidates → `k_sweep.csv` |
| `test_model.py` | Hand-solvable toy tests (5 customers × 2 sites), one per constraint; writes `toy_model.lp` |
| `chennai_customers_10000.csv`, `chennai_sites_50.csv` | The generated 10,000 / 50 dataset (for reference) |

## Run locally
```
pip install -r requirements.txt
python test_model.py        # 8/8 toy tests should pass
python k_sweep.py           # optional: regenerate k_sweep.csv
streamlit run app.py
```

## Model
max Σᵢⱼ (r − c·dᵢⱼ)·xᵢⱼ − Σⱼ Fⱼ·yⱼ  (₹/day)

s.t. Σⱼ xᵢⱼ ≤ Dᵢ (demand) · Σᵢ xᵢⱼ ≤ Cⱼ·yⱼ (capacity) · xᵢⱼ = 0 if dᵢⱼ > Dmax (**max distance**) ·
Σⱼ yⱼ ≤ K (store limit) · xᵢⱼ ≤ Dᵢ·yⱼ (tightening) · yⱼ ∈ {0,1}, xᵢⱼ ≥ 0

## Solution method
1. **Fine grouping of nearby customers:** customers in the same grid square (side = Dmax/10, e.g. 300 m at 3 km)
   that can reach exactly the same set of stores become one demand point with their total demand.
   - Point-to-store distance = demand-weighted average of its customers' real distances, so the delivery cost is exact.
   - Grouping never mixes customers with different reachable stores, so the max-distance rule holds for every customer.
   - Tested against the un-grouped model: profit within ~0.03 %, 5–10× faster.
2. **Warm start:** a greedy heuristic (add stores one at a time, then swap/drop) gives SCIP a strong starting solution.
3. **Optimality-gap control** (sidebar): 0 = prove optimal; a small % stops once the answer is proven within it.

## Key assumptions
- Customers: 80 % clustered around 16 Chennai neighbourhoods, 20 % city-wide; demand 1/2/2.5/3/5 orders/day in the
  proportions of the class sample data.
- Candidates: demand-weighted K-Means centres. Capacity 1,500–2,000 orders/day; fixed cost ₹15,000 + ₹4×capacity per day (±10 %).
- Margin ₹90/order, delivery ₹8/km/order, road distance = haversine × 1.3, default Dmax 3 km.
