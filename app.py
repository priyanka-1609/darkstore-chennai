"""
Dark Store Location Optimisation - Chennai
Streamlit front-end.  Run locally:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import pydeck as pdk
import altair as alt
import streamlit as st

from data import generate_customers, make_candidates, ROAD_FACTOR, inr, lakh
from model import solve_grouped

st.set_page_config(page_title="Dark Store Optimiser - Chennai", page_icon="🏪", layout="wide")

PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (188, 189, 34), (23, 190, 207), (174, 199, 232),
    (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (219, 219, 141), (158, 218, 229), (57, 59, 121), (99, 121, 57),
]
UNSERVED = (120, 120, 120)


# ------------------------------------------------------------------ cached helpers
@st.cache_data(show_spinner=False)
def get_data(n_customers, n_sites, seed):
    cust = generate_customers(n_customers, seed=seed)
    sites = make_candidates(cust, n_sites, seed=seed)
    return cust, sites


@st.cache_data(show_spinner=False)
def run_model(n_customers, n_sites, seed, rev, cpk, max_dist, max_stores, time_limit, gap_pct):
    cust, sites = get_data(n_customers, n_sites, seed)
    kpi, s, c, r = solve_grouped(cust, sites, rev, cpk, max_dist, max_stores,
                                 time_limit_s=time_limit, gap_pct=gap_pct)
    summary = {k: r[k] for k in ("profit", "n_open", "served_pct", "status", "gap_pct", "optimal", "within_target")}
    return kpi, s, c, summary


# ------------------------------------------------------------------ sidebar inputs
st.sidebar.header("Inputs")
with st.sidebar.form("inputs"):
    n_sites = st.number_input("Dark store candidates (count)", 5, 50, 50, step=5)
    n_customers = st.number_input("Number of customers", 200, 21000, 10000, step=500)
    max_dist = st.slider("Max distance store → customer (km)", 1.0, 12.0, 3.0, 0.5,
                         help="Road distance. A customer can only be served by a store within this distance.")
    max_stores = st.number_input("Max stores allowed to open (K)", 1, 50, 25,
                                 help="Upper limit; the model may open fewer if extra stores are not profitable.")
    with st.expander("Economic assumptions"):
        rev = st.number_input("Gross margin per order (Rs)", 10, 500, 90, step=5)
        cpk = st.number_input("Delivery cost per km per order (Rs)", 1, 50, 8)
        time_limit = st.number_input("Solver time limit (s)", 10, 600, 600, step=10)
        gap_pct = st.number_input("Accept solution within (% of optimal)", 0.0, 5.0, 0.0, step=0.1,
                                  help="0 = prove the exact optimum. A small value such as 0.5-1 % lets large "
                                       "cases (big distance / many customers) finish much faster, with a guarantee "
                                       "that the profit is within this % of the best possible.")
        seed = st.number_input("Random seed (data)", 0, 9999, 42)
    submitted = st.form_submit_button("Run optimisation", type="primary", width="stretch")

st.sidebar.caption("Candidate count, customers and seed change the generated data; "
                   "Dmax, K and costs change only the optimisation.")

# ------------------------------------------------------------------ header
st.title("🏪 Dark Store Location Optimisation — Chennai")
st.markdown(
    "Choose which candidate dark stores to open and which store serves each customer to **maximise daily profit**, "
    "subject to store capacity, a **maximum delivery distance**, and a limit on the number of stores. "
    "Solved as a mixed-integer linear program with Google OR-Tools (SCIP)."
)

tab_sol, tab_k, tab_model = st.tabs(["📍 Solution", "📈 Profit vs number of stores", "🧮 Model & assumptions"])

# ------------------------------------------------------------------ run
params = (int(n_customers), int(n_sites), int(seed), float(rev), float(cpk),
          float(max_dist), int(min(max_stores, n_sites)), int(time_limit), float(gap_pct))
if submitted or "params" not in st.session_state:
    st.session_state["params"] = params
p = st.session_state["params"]

with st.spinner("Generating data and solving the model…"):
    cust, sites = get_data(p[0], p[1], p[2])
    kpi, stores, custs, summ = run_model(*p)

# ------------------------------------------------------------------ SOLUTION TAB
with tab_sol:
    if params != p:
        st.info("Inputs changed — press **Run optimisation** to update the results.")
    g = summ["gap_pct"]
    gap_txt = "unknown" if not np.isfinite(g) else (f"{g:.2f}%" if g < 100 else "very large (>100%)")
    if summ["within_target"] and not summ["optimal"]:
        st.success(f"**Near-optimal solution:** profit is proven to be within {g:.2f}% of the best possible "
                   f"(your target: {p[8]}%).")
    elif not summ["optimal"] and (summ["n_open"] == 0 or not np.isfinite(g) or g >= 100):
        st.error(f"**The solver ran out of time before finding a good solution** (gap {gap_txt}). "
                 f"{inr(p[0])} customers with a {p[5]} km radius makes the model too large to solve in {p[7]} s. "
                 "These results are NOT meaningful. Allow a small gap (e.g. 1 %) under *Economic assumptions*, "
                 "reduce the max distance or number of customers, or increase the solver time limit.")
    elif not summ["optimal"]:
        st.warning(f"Solver status: {summ['status']}. Gap {gap_txt} — the solution is good but not proven optimal. "
                   "Try a smaller distance or a longer time limit for a proven optimum.")
    if summ["n_open"] == 0 and summ["optimal"]:
        need = sites.fixed_cost_per_day.min() / max(p[3] - p[4] * p[5] / 2, 1)
        st.info(f"**No store is worth opening.** A store needs roughly {inr(need)}+ orders/day within {p[5]} km "
                f"to cover its fixed cost, but with {inr(p[0])} customers spread across Chennai no candidate reaches "
                "that much demand. Increase the number of customers, the max distance, or the margin per order.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net profit / day", f"₹{inr(summ['profit'])}")
    c1.caption(f"= ₹{lakh(summ['profit'])} per day")
    c2.metric("Stores opened", f"{summ['n_open']} / {p[1]}")
    c3.metric("Demand served", f"{summ['served_pct']:.1f}%")
    c4.metric("Max distance", f"{p[5]} km")

    left, right = st.columns([1.3, 1.5])
    with left:
        st.subheader("KPIs")
        st.dataframe(kpi, hide_index=True, width="stretch", height=740,
                     column_config={"KPI": st.column_config.TextColumn(width=290),
                                    "Value": st.column_config.TextColumn(width=240)})

    with right:
        st.subheader("Solution map")
        o1, o2, o3 = st.columns(3)
        show_radius = o1.checkbox("Dmax circles", True)
        show_lines = o2.checkbox("Assignment lines", False)
        show_closed = o3.checkbox("Closed candidates", True)

        open_idx = np.flatnonzero(stores["open"].to_numpy())
        color_of = {j: PALETTE[k % len(PALETTE)] for k, j in enumerate(open_idx)}

        cmap = custs.copy()
        cmap["color"] = [list(color_of.get(j, UNSERVED)) + [200 if j >= 0 else 110] for j in cmap.store_idx]
        cmap["radius"] = 25 + 12 * cmap.points
        cmap["tip"] = (
            "Customer " + cmap.id.astype(str) + "<br/>Demand: " + cmap.points.astype(str) + " orders/day"
            + "<br/>Served: " + cmap.orders_served.round(1).astype(str) + " (" + cmap.status + ")"
            + "<br/>Store: " + cmap.store
            + np.where(cmap.store_idx >= 0, " · " + cmap.km_to_store.round(2).astype(str) + " km", "")
        )

        so = stores[stores.open].copy()
        so["color"] = [list(color_of[j]) + [255] for j in so.index]
        so["tip"] = (
            "<b>" + so.site_id + " (OPEN)</b><br/>Capacity: " + so.capacity_per_day.map(inr)
            + "<br/>Orders served: " + so.orders_served.map(inr)
            + " (" + so.utilisation_pct.astype(str) + "%)<br/>Customers: " + so.customers.map(inr)
            + "<br/>Fixed cost: ₹" + so.fixed_cost_per_day.map(inr)
            + "/day<br/>Profit: ₹" + so.profit_per_day.map(inr) + "/day"
        )
        sc = stores[~stores.open].copy()
        sc["tip"] = ("<b>" + sc.site_id + " (closed)</b><br/>Capacity: " + sc.capacity_per_day.map(inr)
                     + "<br/>Fixed cost: ₹" + sc.fixed_cost_per_day.map(inr) + "/day")

        layers = []
        if show_radius and len(so):
            layers.append(pdk.Layer(
                "ScatterplotLayer", so, get_position=["long", "lat"],
                get_radius=p[5] / ROAD_FACTOR * 1000,          # road km -> straight-line metres
                filled=True, stroked=True, get_fill_color=[0, 0, 0, 0],
                get_line_color="color", line_width_min_pixels=1.5, pickable=False))
        if show_lines:
            served = cmap[cmap.store_idx >= 0].copy()
            served["s_long"] = stores.long.to_numpy()[served.store_idx]
            served["s_lat"] = stores.lat.to_numpy()[served.store_idx]
            layers.append(pdk.Layer(
                "LineLayer", served, get_source_position=["long", "lat"],
                get_target_position=["s_long", "s_lat"], get_color="color", get_width=1, opacity=0.25))
        layers.append(pdk.Layer(
            "ScatterplotLayer", cmap, get_position=["long", "lat"], get_radius="radius",
            get_fill_color="color", radius_min_pixels=1.5, radius_max_pixels=6, pickable=True))
        if show_closed and len(sc):
            layers.append(pdk.Layer(
                "ScatterplotLayer", sc, get_position=["long", "lat"], get_radius=180,
                get_fill_color=[255, 255, 255, 230], get_line_color=[40, 40, 40], stroked=True,
                line_width_min_pixels=2, radius_min_pixels=4, pickable=True))
        if len(so):
            layers.append(pdk.Layer(
                "ScatterplotLayer", so, get_position=["long", "lat"], get_radius=320,
                get_fill_color="color", get_line_color=[0, 0, 0], stroked=True,
                line_width_min_pixels=2.5, radius_min_pixels=7, pickable=True))
            layers.append(pdk.Layer(
                "TextLayer", so, get_position=["long", "lat"], get_text="site_id", get_size=12,
                get_color=[0, 0, 0], get_pixel_offset=[0, -18], font_weight=700))

        view = pdk.ViewState(latitude=float(cust.lat.mean()), longitude=float(cust.long.mean()), zoom=10.3)
        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                                 tooltip={"html": "{tip}", "style": {"fontSize": "12px"}}),
                        height=640, width="stretch")
        st.caption("Large outlined dots = open dark stores (customers share their colour). "
                   "White dots = candidates not opened. Grey dots = unserved customers. "
                   "Circles show the Dmax service radius. Hover for details.")

    st.subheader("Open dark stores")
    cols = ["site_id", "lat", "long", "capacity_per_day", "orders_served", "utilisation_pct", "customers",
            "avg_km", "max_km", "fixed_cost_per_day", "margin", "delivery_cost", "profit_per_day"]
    tbl = stores.loc[stores.open, cols].sort_values("orders_served", ascending=False).round({"lat": 4, "long": 4})
    for col in ["capacity_per_day", "orders_served", "customers", "fixed_cost_per_day", "margin", "delivery_cost",
                "profit_per_day"]:
        tbl[col] = tbl[col].map(inr)                      # Indian number format (12,34,567)
    st.dataframe(tbl.rename(columns={
        "site_id": "Store", "capacity_per_day": "Capacity (orders/day)", "orders_served": "Orders served",
        "utilisation_pct": "Utilisation (%)", "customers": "Customers", "avg_km": "Avg km", "max_km": "Max km",
        "fixed_cost_per_day": "Fixed cost (₹/day)", "margin": "Margin (₹/day)", "delivery_cost": "Delivery cost (₹/day)",
        "profit_per_day": "Profit (₹/day)"}), hide_index=True, width="stretch")

    d1, d2 = st.columns(2)
    d1.download_button("⬇ Customer assignments (CSV)", custs.to_csv(index=False), "customer_assignments.csv")
    d2.download_button("⬇ All candidate stores (CSV)", stores.to_csv(index=False), "stores.csv")

# ------------------------------------------------------------------ PROFIT vs K TAB
with tab_k:
    st.subheader("Profit when at most K dark stores may open — 10,000 customers, 50 candidates")
    st.markdown("K = 2, 5, 10, 20, 25. Margin ₹90/order, delivery ₹8/km/order, seed 42. "
                "All 20 runs solved to proven optimality (Solver status column).")

    def k_chart(df):
        df = df.copy()
        df["profit_lakh"] = df.profit_per_day / 1e5
        df["Profit"] = "₹" + df.profit_per_day.map(inr) + "/day"
        base = alt.Chart(df).encode(x=alt.X("K:O", title="Max stores allowed (K)"))
        line = base.mark_line(point=alt.OverlayMarkDef(size=80)).encode(
            y=alt.Y("profit_lakh:Q", title="Net profit (₹ lakh / day)", axis=alt.Axis(format=".1f")),
            color=alt.Color("Dmax:N", title="Max distance"),
            tooltip=["Dmax", "K", "Profit", "stores_opened", "demand_served_pct", "status"])
        text = base.mark_text(dy=-12, fontSize=11).encode(
            y="profit_lakh:Q", color="Dmax:N", text="stores_opened:Q")
        return (line + text).properties(height=420)

    if os.path.exists("k_sweep.csv"):
        ks = pd.read_csv("k_sweep.csv")
        ks["Dmax"] = ks.max_dist_km.astype(str) + " km"
        st.altair_chart(k_chart(ks), width="stretch")
        st.caption("Labels on points = stores actually opened. Pre-computed with `k_sweep.py`.")
        ks_tbl = ks.drop(columns="Dmax").copy()
        ks_tbl["profit_per_day"] = ks_tbl.profit_per_day.map(inr)
        st.dataframe(ks_tbl.rename(columns={
            "max_dist_km": "Dmax (km)", "profit_per_day": "Profit (₹/day)", "stores_opened": "Stores opened",
            "demand_served_pct": "Demand served (%)", "status": "Solver status", "gap_pct": "Gap (%)",
            "solve_s": "Time (s)"}).round({"Demand served (%)": 1, "Gap (%)": 3, "Time (s)": 1}),
            hide_index=True, width="stretch")
        st.markdown(
            "**Reading the curve.** With few stores allowed, every store is filled close to capacity and each extra "
            "store adds a large block of profitable orders, so profit rises steeply. Once enough stores are open to "
            "serve all demand reachable within Dmax, an additional store only shortens some deliveries, which saves "
            "less than its fixed cost — the model stops opening stores even though K allows more, and the curve "
            "flattens. A larger Dmax lets each store reach more customers, so profit is higher and the plateau "
            "comes at a different number of stores.")

    st.divider()
    st.markdown(f"**Run the sweep live** with the current sidebar settings "
                f"({inr(p[0])} customers, {p[1]} candidates, Dmax {p[5]} km). Takes up to a few minutes.")
    if st.button("Run K = 2, 5, 10, 20, 25 now"):
        rows = []
        bar = st.progress(0.0)
        for n, K in enumerate([2, 5, 10, 20, 25]):
            _, _, _, s_ = run_model(p[0], p[1], p[2], p[3], p[4], p[5], min(K, p[1]), p[7], p[8])
            rows.append(dict(K=K, profit_per_day=round(s_["profit"]), stores_opened=s_["n_open"],
                             demand_served_pct=round(s_["served_pct"], 1), status=s_["status"],
                             Dmax=f"{p[5]} km"))
            bar.progress((n + 1) / 5)
        live = pd.DataFrame(rows)
        st.altair_chart(k_chart(live), width="stretch")
        st.dataframe(live.assign(profit_per_day=live.profit_per_day.map(inr)), hide_index=True, width="stretch")

# ------------------------------------------------------------------ MODEL TAB
with tab_model:
    st.subheader("Mathematical model")
    st.markdown("**Sets:** $I$ customers (demand locations), $J$ candidate dark-store sites.")
    st.markdown("**Parameters:** $D_i$ daily orders, $C_j$ daily capacity, $F_j$ fixed cost (₹/day), "
                "$d_{ij}$ road distance (km), $r$ margin per order (₹), $c$ delivery cost (₹/km/order), "
                "$D_{max}$ max distance, $K$ max stores.")
    st.markdown("**Decision variables:** $y_j \\in \\{0,1\\}$ open store $j$; $x_{ij} \\ge 0$ orders/day from $j$ to $i$.")
    st.latex(r"\max \; \sum_{i\in I}\sum_{j\in J} (r - c\, d_{ij})\, x_{ij} \;-\; \sum_{j\in J} F_j\, y_j")
    st.latex(r"""\begin{aligned}
    &\textstyle\sum_{j} x_{ij} \le D_i && \forall i && \text{(demand; unmet demand allowed)}\\
    &\textstyle\sum_{i} x_{ij} \le C_j\, y_j && \forall j && \text{(capacity)}\\
    &x_{ij} = 0 \quad \text{if } d_{ij} > D_{max} && \forall i,j && \text{(max distance — enhancement)}\\
    &\textstyle\sum_{j} y_j \le K && && \text{(store limit)}\\
    &x_{ij} \le D_i\, y_j && \forall i,j && \text{(valid inequality, speeds up the solver)}
    \end{aligned}""")
    st.markdown("""
**Changes from the base model in class**
- *Max-distance constraint (assignment enhancement):* $x_{ij}$ is only created when $d_{ij} \\le D_{max}$; this also shrinks
  the model (e.g. 500,000 → ~15,000 assignment variables at 3 km).
- *Demand ≤ instead of =, with revenue in the objective:* with a distance limit or few stores some customers cannot be served,
  and the equality version would be infeasible (class slide "Dark store location optimisation: revisited").
- *All terms in ₹/day:* fixed cost is expressed per day, delivery cost = ₹/km × km × orders, so the objective adds like units.
- *Store limit K:* needed for the profit-vs-K analysis.
- *Tightening constraint $x_{ij} \\le D_i y_j$:* implied by the capacity constraint for integer $y$, but it makes the LP
  relaxation much tighter; without it SCIP could not prove optimality within 2 minutes at 10,000 × 50.
- $x_{ij}$ is continuous: for fixed $y$ the remaining problem is a transportation problem with integer data, so optimal
  flows are integral anyway.

**Assumptions**
| Item | Value | Basis |
|---|---|---|
| Geography | Chennai, lat 12.85–13.23, coastline-clipped | assignment |
| Customers | 80 % clustered around 16 neighbourhoods (T. Nagar, Anna Nagar, Velachery, Adyar, OMR …), 20 % spread city-wide | dense residential demand |
| Demand per customer | 1, 2, 2.5, 3 or 5 orders/day, in the proportions of the class sample data (`points`) | class data |
| Candidate sites | demand-weighted K-Means centres of customers | slide 25: dense residential areas |
| Capacity | 1,500–2,000 orders/day (bigger where local demand is higher) | slide 25 |
| Fixed cost | ₹15,000 + ₹4 × capacity per day, ±10 % (≈ ₹19–25k/day ≈ ₹6–7.5 lakh/month: rent, staff, utilities) | assumption |
| Margin per order | ₹90 (e.g. ₹450 basket × 20 %) | assumption |
| Delivery cost | ₹8 per km per order (rider + fuel) | assumption |
| Distance | haversine × 1.3 road factor | assumption |
| Dmax | default 3 km, range 1–12 km (class: 1.7–2 km service area for dense demand) | slide 25 |
| Solver | OR-Tools + SCIP with a heuristic warm start; time limit and optimality-gap target adjustable | class (MIP controls) |

**Solution method (speed-ups)**
1. *Grouping nearby customers (as in the class reference code, but much finer):* customers in the same small grid square
   (side = Dmax / 10, e.g. 300 m at 3 km) **that can reach exactly the same set of stores** are merged into one demand
   point carrying their total demand. Safeguards so the answer stays accurate:
   - the delivery distance of a point to a store is the demand-weighted **average of its customers' real distances**,
     so delivery cost and profit equal the real cost of serving those customers;
   - customers are only merged if every one of them is within Dmax of the same stores, so the **max-distance
     constraint holds exactly for every real customer**.
   Tested against the un-grouped model: profit differs by about 0.03 % while solving 5–10× faster.
2. *Warm start:* a greedy heuristic (add the most profitable store one at a time, then try swapping/dropping stores)
   picks a set of stores in seconds; the best assignment for that set is found with an LP and given to SCIP as a
   starting point (hint). SCIP then solves the model exactly from that strong start.
3. *Optimality-gap control (class slide "MIP controls"):* gap = 0 proves the exact optimum of the model. A small target
   (e.g. 1 %) stops as soon as the solution is *proven* within that % of the best possible profit.
4. Distances above ≈ 11 km are never profitable with ₹90 margin and ₹8/km delivery (90 / 8 = 11.25 km), so the
   distance slider goes up to 12 km.

**Validation:** `test_model.py` contains hand-solvable toy cases (5 customers, 2 sites) that check each constraint
(max distance, capacity, fixed cost, store limit) — all pass; the toy model is exported to `toy_model.lp`.
""")
