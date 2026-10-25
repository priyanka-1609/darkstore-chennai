"""
Dark-store location optimisation model (OR-Tools + SCIP).

Base model (class slides 26-28): capacitated facility location with revenue.
Enhancements for Assignment 1:
  * Max-distance constraint: customer i can be served by store j only if d_ij <= Dmax
    (x_ij is simply not created for farther pairs, i.e. x_ij = 0).
  * Store-count limit: sum_j y_j <= K.
  * Demand constraint relaxed to <= D_i (slide 28) so the model stays feasible when
    Dmax / K / capacity make it impossible to serve everybody.

    max  sum_ij (rev - cpk * d_ij) * x_ij  -  sum_j F_j * y_j          [Rs/day]
    s.t. sum_j x_ij        <= D_i              for all i              (demand)
         sum_i x_ij        <= C_j * y_j        for all j              (capacity)
         x_ij              <= D_i * y_j        for all valid (i,j)    (tightening, speeds up B&B)
         sum_j y_j         <= K                                       (store limit)
         x_ij = 0 if d_ij > Dmax                                      (max distance)
         y_j in {0,1}, x_ij >= 0
"""
import time
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from data import inr, lakh

STATUS = {
    pywraplp.Solver.OPTIMAL: "Optimal",
    pywraplp.Solver.FEASIBLE: "Feasible (time limit hit, not proven optimal)",
    pywraplp.Solver.INFEASIBLE: "Infeasible",
    pywraplp.Solver.UNBOUNDED: "Unbounded",
    pywraplp.Solver.ABNORMAL: "Abnormal",
    pywraplp.Solver.NOT_SOLVED: "Not solved",
}


# ---------------------------------------------------------------------------------
# Speed-up: a fast heuristic finds a good starting solution, which is handed to SCIP
# as a "hint" (warm start). SCIP still solves the full exact model; it just starts
# from a strong incumbent instead of from scratch. Optionally, an LP bound from PDLP
# (a fast first-order LP solver) certifies how far the heuristic can be from optimal.
# ---------------------------------------------------------------------------------
def _greedy_heuristic(D, F, C, W, K, time_budget_s=8.0):
    """Greedy add + swap/drop local search on an approximate profit.

    W[i, j] = profit if customer i is fully served by store j (0 if not allowed).
    Approximation of capacity: each customer goes to its best open store; an overloaded
    store serves its customers proportionally. Returns a boolean mask of open stores.
    """
    n, m = W.shape
    K = int(min(K, m))

    def score(best, owner):
        srv = best > 0
        load = np.bincount(owner[srv], weights=D[srv], minlength=m)
        scale = np.minimum(1.0, C / np.maximum(load, 1e-9))
        return float((best * srv * scale[np.clip(owner, 0, None)]).sum())

    def full(mask):
        if not mask.any():
            return 0.0, np.zeros(n), np.full(n, -1)
        idx = np.flatnonzero(mask)
        Wo = W[:, idx]
        a = Wo.argmax(axis=1)
        best = Wo[np.arange(n), a]
        owner = idx[a]
        return score(best, owner) - F[mask].sum(), best, owner

    S = np.zeros(m, bool)
    val, best, owner = 0.0, np.zeros(n), np.full(n, -1)
    for _ in range(K):                                       # greedy add
        fixed = F[S].sum()
        cand_val, cand_j = -np.inf, -1
        for j in np.flatnonzero(~S):
            better = W[:, j] > best
            v = score(np.where(better, W[:, j], best), np.where(better, j, owner)) - fixed - F[j]
            if v > cand_val:
                cand_val, cand_j = v, j
        if cand_j < 0 or cand_val <= val + 1e-6:
            break
        S[cand_j] = True
        val, best, owner = full(S)

    t0 = time.time()                                         # swap / drop improvement
    improved = True
    while improved and time.time() - t0 < time_budget_s:
        improved = False
        for a in np.flatnonzero(S):
            for b in [None] + list(np.flatnonzero(~S)):
                T = S.copy(); T[a] = False
                if b is not None:
                    T[b] = True
                v = full(T)[0]
                if v > val + 1e-6:
                    S, val, improved = T, v, True
                    break
            if improved or time.time() - t0 >= time_budget_s:
                break
    return S


def _transport_lp(open_mask, D, C, unit_profit, ii, jj):
    """Exact best assignment for a FIXED set of open stores (an LP).
    Returns (profit before fixed cost, x values aligned with (ii, jj))."""
    s = pywraplp.Solver.CreateSolver("GLOP")
    ks = np.flatnonzero(open_mask[jj])
    xv = np.zeros(len(ii))
    if len(ks) == 0:
        return 0.0, xv
    x = [s.NumVar(0.0, D[ii[k]], "") for k in ks]
    dem, cap = {}, {}
    obj = s.Objective()
    for v, k in zip(x, ks):
        i, j = ii[k], jj[k]
        if i not in dem:
            dem[i] = s.Constraint(-s.infinity(), D[i])
        if j not in cap:
            cap[j] = s.Constraint(-s.infinity(), C[j])
        dem[i].SetCoefficient(v, 1.0)
        cap[j].SetCoefficient(v, 1.0)
        obj.SetCoefficient(v, float(unit_profit[i, j]))
    obj.SetMaximization()
    s.Solve()
    xv[ks] = [v.solution_value() for v in x]
    return obj.Value(), xv


def _lp_bound_pdlp(D, F, C, unit_profit, ii, jj, max_stores, time_limit_s):
    """Upper bound on the optimal profit: LP relaxation (y continuous) solved with PDLP."""
    s = pywraplp.Solver.CreateSolver("PDLP")
    if s is None:
        return np.nan
    s.SetTimeLimit(int(time_limit_s * 1000))
    m = len(F)
    y = [s.NumVar(0.0, 1.0, "") for _ in range(m)]
    x = [s.NumVar(0.0, D[i], "") for i in ii]
    dem = {i: s.Constraint(-s.infinity(), D[i]) for i in np.unique(ii)}
    cap = [s.Constraint(-s.infinity(), 0.0) for _ in range(m)]
    for j in range(m):
        cap[j].SetCoefficient(y[j], -C[j])
    obj = s.Objective()
    for k, (i, j) in enumerate(zip(ii, jj)):
        dem[i].SetCoefficient(x[k], 1.0)
        cap[j].SetCoefficient(x[k], 1.0)
        link = s.Constraint(-s.infinity(), 0.0)
        link.SetCoefficient(x[k], 1.0)
        link.SetCoefficient(y[j], -D[i])
        obj.SetCoefficient(x[k], float(unit_profit[i, j]))
    lim = s.Constraint(-s.infinity(), float(max_stores))
    for j in range(m):
        lim.SetCoefficient(y[j], 1.0)
        obj.SetCoefficient(y[j], -F[j])
    obj.SetMaximization()
    if s.Solve() != pywraplp.Solver.OPTIMAL:
        return np.nan
    return obj.Value() * (1 + 1e-4) + 1.0        # small safety margin for PDLP's tolerance


def build_and_solve(D, F, C, d, rev, cpk, max_dist, max_stores,
                    time_limit_s=60, gap_pct=0.0, solver_name="SCIP", lp_file=None,
                    warm_start=True, d_feas=None):
    """Core MILP. D (n), F (m), C (m), d (n x m) are NumPy arrays.

    gap_pct    : stop once the solution is proven within this % of optimal (0 = prove optimal).
    warm_start : run the greedy heuristic first and give its solution to SCIP as a hint.
    Returns dict with status, objective, y (m,), x as (ii, jj, values), and solver stats.
    """
    t0 = time.time()
    D = np.asarray(D, float); F = np.asarray(F, float)
    C = np.asarray(C, float); d = np.asarray(d, float)
    n, m = d.shape
    unit_profit = rev - cpk * d
    # max-distance constraint: only create x_ij for pairs within Dmax
    # (pairs that could never make money are also skipped - they would always be 0)
    d_check = d if d_feas is None else np.asarray(d_feas, float)   # distance used for the Dmax rule
    valid = (d_check <= max_dist) & (unit_profit > 0)
    ii, jj = np.nonzero(valid)

    # ---- 1. heuristic starting solution
    h_y, h_x, h_val, t_heur = None, None, -np.inf, 0.0
    pdlp_bound = np.nan
    if warm_start and max_stores > 0 and len(ii):
        W = np.where(valid, unit_profit * D[:, None], 0.0)
        S = _greedy_heuristic(D, F, C, W, max_stores)
        flow_profit, h_x = _transport_lp(S, D, C, unit_profit, ii, jj)
        h_val = flow_profit - F[S].sum()
        if h_val < 0:                                   # opening nothing is always feasible
            S = np.zeros(m, bool); h_x = np.zeros(len(ii)); h_val = 0.0
        h_y = S.astype(float)
        t_heur = time.time() - t0

        # ---- 2. large model: fast LP bound (PDLP) so the reported gap is meaningful even if SCIP
        #         times out; if a gap is allowed and the heuristic is already within it, stop here.
        if len(ii) > 15000:
            bound = _lp_bound_pdlp(D, F, C, unit_profit, ii, jj, max_stores,
                                   max(5.0, 0.5 * (time_limit_s - (time.time() - t0))))
            pdlp_bound = bound
            if gap_pct > 0 and np.isfinite(bound) and h_val > 0 and (bound - h_val) / h_val * 100 <= gap_pct:
                g = (bound - h_val) / h_val * 100
                return dict(status=f"Near-optimal: proven within {g:.2f}% (heuristic + LP bound)",
                            optimal=False, within_target=True, objective=h_val, best_bound=bound,
                            gap_pct=g, y=h_y, ii=ii, jj=jj, xv=h_x,
                            n_vars=m + len(ii), n_cons=len(np.unique(ii)) + m + len(ii) + 1,
                            build_s=time.time() - t0, solve_s=0.0, heuristic_s=t_heur)

    # ---- 3. exact MILP with SCIP (warm-started with the heuristic solution)
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Solver {solver_name} not available")
    solver.SetTimeLimit(int(max(1.0, time_limit_s - (time.time() - t0)) * 1000))
    if gap_pct > 0 and solver_name == "SCIP":
        solver.SetSolverSpecificParametersAsString(f"limits/gap = {gap_pct / 100}")

    # ---- Decision variables
    y = [solver.BoolVar(f"y_{j}") for j in range(m)]
    x = [solver.NumVar(0.0, D[i], f"x_{i}_{j}") for i, j in zip(ii, jj)]

    # ---- Constraints
    # (customers with no store within Dmax get no demand constraint - they have no x_ij)
    demand = {i: solver.Constraint(-solver.infinity(), D[i], f"demand_{i}") for i in np.unique(ii)}
    capacity = [solver.Constraint(-solver.infinity(), 0.0, f"capacity_{j}") for j in range(m)]
    for j in range(m):
        capacity[j].SetCoefficient(y[j], -C[j])                 # sum_i x_ij - C_j y_j <= 0
    for k, (i, j) in enumerate(zip(ii, jj)):
        demand[i].SetCoefficient(x[k], 1.0)
        capacity[j].SetCoefficient(x[k], 1.0)
        link = solver.Constraint(-solver.infinity(), 0.0, f"link_{i}_{j}")   # x_ij - D_i y_j <= 0
        link.SetCoefficient(x[k], 1.0)
        link.SetCoefficient(y[j], -D[i])
    limit = solver.Constraint(-solver.infinity(), float(max_stores), "store_limit")
    for j in range(m):
        limit.SetCoefficient(y[j], 1.0)

    # ---- Objective: maximise daily profit
    obj = solver.Objective()
    for k, (i, j) in enumerate(zip(ii, jj)):
        obj.SetCoefficient(x[k], float(unit_profit[i, j]))
    for j in range(m):
        obj.SetCoefficient(y[j], -F[j])
    obj.SetMaximization()

    if lp_file:
        with open(lp_file, "w") as f:
            f.write(solver.ExportModelAsLpFormat(False))

    if h_y is not None:
        solver.SetHint(y + x, list(h_y) + list(h_x))      # warm start

    t_build = time.time() - t0
    status = solver.Solve()
    t_solve = time.time() - t0 - t_build

    has_sol = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    yv = np.array([v.solution_value() for v in y]) if has_sol else np.zeros(m)
    xv = np.array([v.solution_value() for v in x]) if has_sol else np.zeros(len(x))
    best_bound = obj.BestBound() if has_sol else np.nan
    objective = obj.Value() if has_sol else np.nan
    status_txt = STATUS.get(status, str(status))
    if h_y is not None and (not has_sol or h_val > objective + 1e-6):
        # SCIP stopped before matching the heuristic: keep the heuristic solution
        yv, xv, objective = h_y, h_x, h_val
        has_sol = True
        if not np.isfinite(best_bound):
            best_bound = np.nan
        status_txt = "Feasible (time limit hit, heuristic solution, not proven optimal)"
    if np.isfinite(pdlp_bound):                         # use the tighter of the two upper bounds
        best_bound = pdlp_bound if not np.isfinite(best_bound) else min(best_bound, pdlp_bound)
    gap = (abs(best_bound - objective) / max(abs(objective), 1e-9) * 100) if np.isfinite(best_bound) else np.nan
    optimal = status == pywraplp.Solver.OPTIMAL and (gap_pct == 0 or (np.isfinite(gap) and gap < 0.01))
    within = status == pywraplp.Solver.OPTIMAL or (np.isfinite(gap) and gap <= gap_pct)

    return dict(status=status_txt if (optimal or status != pywraplp.Solver.OPTIMAL)
                else f"Near-optimal: proven within {gap:.2f}% (target {gap_pct}%)",
                optimal=optimal, within_target=within,
                objective=objective, best_bound=best_bound, gap_pct=gap,
                y=yv, ii=ii, jj=jj, xv=xv,
                n_vars=solver.NumVariables(), n_cons=solver.NumConstraints(),
                build_s=t_build, solve_s=t_solve, heuristic_s=t_heur)


def solve(cust, sites, d, rev, cpk, max_dist, max_stores, time_limit_s=60, gap_pct=0.0, lp_file=None,
          warm_start=True, d_feas=None):
    """Solve and summarise. cust: id, lat, long, points; sites: site_id, lat, long,
    capacity_per_day, fixed_cost_per_day; d: (n x m) road km.

    Returns (kpi_df, store_df, cust_df, raw).
    """
    D = cust["points"].to_numpy(float)
    F = sites["fixed_cost_per_day"].to_numpy(float)
    C = sites["capacity_per_day"].to_numpy(float)
    res = build_and_solve(D, F, C, d, rev, cpk, max_dist, max_stores, time_limit_s, gap_pct,
                          lp_file=lp_file, warm_start=warm_start, d_feas=d_feas)

    n, m = len(cust), len(sites)
    is_open = res["y"] > 0.5
    keep = res["xv"] > 1e-6
    flows = pd.DataFrame({"i": res["ii"][keep], "j": res["jj"][keep], "orders": res["xv"][keep]})
    flows["km"] = d[flows.i, flows.j]
    flows["margin"] = rev * flows.orders
    flows["delivery_cost"] = cpk * flows.km * flows.orders

    # ---- per customer (main store = the one sending the most orders)
    main = flows.sort_values("orders", ascending=False).drop_duplicates("i").set_index("i")
    served = flows.groupby("i").orders.sum().reindex(range(n), fill_value=0).to_numpy()
    out_c = cust.copy().reset_index(drop=True)
    out_c["orders_served"] = served.round(3)
    out_c["store_idx"] = main.j.reindex(range(n)).fillna(-1).astype(int).to_numpy()
    out_c["store"] = np.where(out_c.store_idx >= 0,
                              sites.site_id.to_numpy()[out_c.store_idx.clip(lower=0)], "Unserved")
    out_c["km_to_store"] = main.km.reindex(range(n)).to_numpy()
    out_c["status"] = np.select([served >= D - 1e-6, served > 1e-6], ["Fully served", "Partly served"], "Unserved")

    # ---- per store
    g = flows.groupby("j")
    out_s = sites.copy().reset_index(drop=True)
    out_s["open"] = is_open
    out_s["customers"] = g.i.nunique().reindex(range(m), fill_value=0).to_numpy()
    out_s["orders_served"] = g.orders.sum().reindex(range(m), fill_value=0).to_numpy().round(1)
    out_s["utilisation_pct"] = np.where(is_open, 100 * out_s.orders_served / out_s.capacity_per_day, np.nan).round(1)
    km_orders = (flows.km * flows.orders).groupby(flows.j).sum()
    out_s["avg_km"] = (km_orders / g.orders.sum()).reindex(range(m)).to_numpy().round(2)   # order-weighted
    out_s["max_km"] = g.km.max().reindex(range(m)).to_numpy().round(2)
    out_s["margin"] = g.margin.sum().reindex(range(m), fill_value=0).to_numpy()
    out_s["delivery_cost"] = g.delivery_cost.sum().reindex(range(m), fill_value=0).to_numpy()
    out_s["profit_per_day"] = np.where(is_open, out_s.margin - out_s.delivery_cost - out_s.fixed_cost_per_day, 0).round(0)

    # ---- KPIs
    tot_orders = flows.orders.sum()
    margin = flows.margin.sum(); dcost = flows.delivery_cost.sum()
    fcost = F[is_open].sum()
    reachable = (d <= max_dist).any(axis=1)
    kpis = [
        ("Net profit (₹/day)", f"₹{inr(margin - dcost - fcost)}  (₹{lakh(margin - dcost - fcost)})"),
        ("Gross margin from orders (₹/day)", f"₹{inr(margin)}"),
        ("Delivery cost (₹/day)", f"₹{inr(dcost)}"),
        ("Fixed cost of open stores (₹/day)", f"₹{inr(fcost)}"),
        ("Dark stores opened", f"{is_open.sum()} of {m} candidates (limit K = {max_stores})"),
        ("Customers", f"{inr(n)}"),
        ("Customers fully / partly served", f"{inr((out_c.status == 'Fully served').sum())} / {inr((out_c.status == 'Partly served').sum())}"),
        ("Customers unserved", f"{inr((out_c.status == 'Unserved').sum())}"),
        ("Customers with no candidate within Dmax", f"{inr((~reachable).sum())}"),
        ("Total demand (orders/day)", f"{inr(D.sum())}"),
        ("Orders served (orders/day)", f"{inr(tot_orders)}"),
        ("Demand served (%)", f"{100 * tot_orders / D.sum():.1f}"),
        ("Avg delivery distance (km, order-weighted)", f"{(flows.km * flows.orders).sum() / tot_orders:.2f}" if tot_orders else "-"),
        ("Max delivery distance (km)", f"{flows.km.max():.2f} (limit {max_dist} km)" if tot_orders else "-"),
        ("Avg capacity utilisation of open stores (%)", f"{np.nanmean(out_s.utilisation_pct[is_open]):.1f}" if is_open.any() else "-"),
        ("Profit per open store (₹/day)", f"₹{inr((margin - dcost - fcost) / is_open.sum())}" if is_open.any() else "-"),
        ("Solver status", res["status"]),
        ("Optimality gap (%)", f"{res['gap_pct']:.3f}" if np.isfinite(res['gap_pct']) else "unknown"),
        ("Model size", f"{inr(res['n_vars'])} variables, {inr(res['n_cons'])} constraints"),
        ("Solve time (s)", f"{res['build_s'] + res['solve_s']:.1f} (incl. {res['heuristic_s']:.1f} s warm-start heuristic)"),
    ]
    kpi_df = pd.DataFrame(kpis, columns=["KPI", "Value"])
    res["profit"] = margin - dcost - fcost
    res["served_pct"] = 100 * tot_orders / D.sum()
    res["n_open"] = int(is_open.sum())
    return kpi_df, out_s, out_c, res



def solve_grouped(cust, sites, rev, cpk, max_dist, max_stores, time_limit_s=60, gap_pct=0.0, seed=42):
    """Speed-up: group nearby customers into demand points on a square grid (cell side = Dmax/10),
    only merging customers that can reach exactly the same set of stores; solve the MILP on the
    points, then map the solution back to every individual customer.

    Accuracy safeguards:
      * cost distance of point g to store j = demand-weighted AVERAGE real distance of its customers,
        so the delivery cost (and profit) in the model equals the real cost of serving those customers;
      * Dmax rule uses the MAXIMUM real distance of its customers, so no real customer is ever
        served from farther than Dmax.
    Returns (kpi_df, store_df, cust_df, raw) like solve().
    """
    from data import group_customers_grid, cell_km_for, distance_matrix

    t0 = time.time()
    cell_km = cell_km_for(max_dist)
    d_cust = distance_matrix(cust, sites)                       # real customer-store road km
    reach = (d_cust <= max_dist) & (rev - cpk * d_cust > 0)     # stores each customer may use
    pts, labels = group_customers_grid(cust, cell_km, reach)
    G, m = len(pts), len(sites)
    Dc = cust["points"].to_numpy(float)
    d_avg = np.zeros((G, m)); np.add.at(d_avg, labels, d_cust * Dc[:, None])
    d_avg /= pts["points"].to_numpy(float)[:, None]
    d_max = np.zeros((G, m)); np.maximum.at(d_max, labels, d_cust)
    t_group = time.time() - t0
    kpi, s_out, p_out, res = solve(pts, sites, d_avg, rev, cpk, max_dist, max_stores, time_limit_s, gap_pct,
                                   d_feas=d_max)

    # ---- map point-level solution back to individual customers (each point's customers share its service)
    frac = (p_out.orders_served / p_out.points).clip(0, 1).to_numpy()[labels]
    c = cust.copy().reset_index(drop=True)
    c["group"] = labels + 1
    c["orders_served"] = (c.points * frac).round(3)
    c["store_idx"] = p_out.store_idx.to_numpy()[labels]
    c["store"] = p_out.store.to_numpy()[labels]
    si = c.store_idx.clip(lower=0).to_numpy()
    c["km_to_store"] = np.where(c.store_idx >= 0, d_cust[np.arange(len(c)), si], np.nan)
    c["status"] = np.select([frac >= 1 - 1e-6, frac > 1e-6], ["Fully served", "Partly served"], "Unserved")
    s_out["customers"] = (c[c.store_idx >= 0].groupby("store_idx").size()
                          .reindex(range(m), fill_value=0).to_numpy())
    s_out["max_km"] = (c[c.store_idx >= 0].groupby("store_idx").km_to_store.max()
                       .reindex(range(m)).to_numpy().round(2))

    # ---- customer-level KPIs from real locations
    served = c.orders_served > 0
    kv = dict(zip(kpi.KPI, kpi.Value))
    kv["Customers"] = f"{inr(len(c))}"
    kv["Customers fully / partly served"] = (f"{inr((c.status == 'Fully served').sum())} / "
                                             f"{inr((c.status == 'Partly served').sum())}")
    kv["Customers unserved"] = f"{inr((c.status == 'Unserved').sum())}"
    kv["Customers with no candidate within Dmax"] = f"{inr((~(d_cust <= max_dist).any(axis=1)).sum())}"
    if served.any():
        kv["Max delivery distance (km)"] = f"{c.km_to_store[served].max():.2f} (limit {max_dist} km)"
    kv["Solve time (s)"] = f"{kv['Solve time (s)']} + {t_group:.1f} s grouping"
    rows = []
    for k_, v_ in kv.items():
        rows.append((k_, v_))
        if k_ == "Customers":
            rows.append(("Demand points in model (grid groups)",
                         f"{inr(G)} groups of nearby customers ({cell_km * 1000:.0f} m grid squares, "
                         f"avg {len(c) / G:.1f} customers each)"))
    res["n_groups"] = G
    res["group_s"] = t_group
    return pd.DataFrame(rows, columns=["KPI", "Value"]), s_out, c, res
