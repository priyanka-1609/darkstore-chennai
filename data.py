"""
Data generation for the Chennai dark-store problem.

Customers : clustered around real Chennai residential/commercial hotspots
            (+ some uniform background demand), kept on land (west of the coast).
            Daily demand per customer uses the tiers of sir's sample data ('points').
Candidates: weighted K-Means centres of customer demand (as in course.py),
            so candidate stores sit where demand is (slide 25: dense residential areas).
Capacity  : 1500-2000 orders/day (slide 25), larger where local demand is higher.
Fixed cost: Rs/day, base + per-unit-capacity, with +/-10 % noise.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---------------- Geography ----------------
LAT_MIN, LAT_MAX = 12.85, 13.23
LON_MIN = 80.08
# Approximate Chennai coastline (lat -> lon of the shore), used to drop points in the sea
COAST_LAT = [12.85, 12.90, 12.95, 13.00, 13.05, 13.10, 13.15, 13.23]
COAST_LON = [80.240, 80.248, 80.258, 80.270, 80.282, 80.293, 80.300, 80.310]

# (name, lat, lon, relative weight, spread in km)
HOTSPOTS = [
    ("T. Nagar",       13.0418, 80.2341, 1.0, 1.5),
    ("Anna Nagar",     13.0850, 80.2101, 1.0, 1.8),
    ("Velachery",      12.9815, 80.2180, 0.9, 1.6),
    ("Adyar",          13.0012, 80.2565, 0.8, 1.3),
    ("Mylapore",       13.0330, 80.2690, 0.7, 1.2),
    ("Vadapalani",     13.0500, 80.2121, 0.7, 1.4),
    ("Guindy",         13.0067, 80.2206, 0.5, 1.3),
    ("Porur",          13.0382, 80.1565, 0.6, 1.8),
    ("Ambattur",       13.1143, 80.1548, 0.6, 2.0),
    ("Perambur",       13.1180, 80.2330, 0.6, 1.6),
    ("Tondiarpet",     13.1260, 80.2880, 0.5, 1.4),
    ("Chromepet",      12.9516, 80.1462, 0.6, 1.8),
    ("Tambaram",       12.9249, 80.1000, 0.6, 2.0),
    ("Thoraipakkam",   12.9400, 80.2340, 0.8, 1.8),
    ("Sholinganallur", 12.9010, 80.2279, 0.7, 1.8),
    ("Medavakkam",     12.9180, 80.1920, 0.5, 1.6),
]
HOTSPOT_SHARE = 0.80           # share of customers near hotspots; rest spread uniformly

# ---------------- Demand (from sir's sample data 'points' column) ----------------
DEMAND_VALUES = [1, 2, 2.5, 3, 5]
DEMAND_PROBS = [0.474, 0.402, 0.070, 0.032, 0.022]

# ---------------- Store assumptions ----------------
CAP_MIN, CAP_MAX = 1500, 2000          # orders/day (slide 25)
FIXED_BASE, FIXED_PER_CAP = 15000, 4   # Rs/day: base + Rs per unit of capacity
FIXED_NOISE = 0.10                     # +/- 10 %

ROAD_FACTOR = 1.3                      # straight-line km -> road km
KM_PER_DEG = 111.0


def _on_land(lat, lon):
    coast = np.interp(lat, COAST_LAT, COAST_LON)
    return (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon < coast - 0.004)


def generate_customers(n=10000, seed=42):
    """Customers clustered around hotspots + uniform background, all on land."""
    rng = np.random.default_rng(seed)
    w = np.array([h[3] for h in HOTSPOTS]); w = w / w.sum()
    lats, lons = [], []
    while sum(len(a) for a in lats) < n:
        m = 2 * n
        k = rng.choice(len(HOTSPOTS), size=m, p=w)
        hs = np.array([h[1:] for h in HOTSPOTS])           # lat, lon, weight, spread_km
        spread_deg = hs[k, 3] / KM_PER_DEG
        lat = hs[k, 0] + rng.normal(0, 1, m) * spread_deg
        lon = hs[k, 1] + rng.normal(0, 1, m) * spread_deg
        bg = rng.random(m) > HOTSPOT_SHARE                  # uniform background points
        lat[bg] = rng.uniform(LAT_MIN, LAT_MAX, bg.sum())
        lon[bg] = rng.uniform(LON_MIN, max(COAST_LON), bg.sum())
        ok = _on_land(lat, lon)
        lats.append(lat[ok]); lons.append(lon[ok])
    lat = np.concatenate(lats)[:n]; lon = np.concatenate(lons)[:n]
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "lat": lat.round(6),
        "long": lon.round(6),
        "points": rng.choice(DEMAND_VALUES, size=n, p=DEMAND_PROBS),   # daily orders
    })


def make_candidates(cust, n_sites=50, seed=42):
    """Candidate dark-store sites = demand-weighted K-Means centres of customers."""
    rng = np.random.default_rng(seed + 1)
    coords = cust[["lat", "long"]].to_numpy()
    weights = cust["points"].to_numpy()
    km = KMeans(n_clusters=n_sites, random_state=seed, n_init=10)
    labels = km.fit_predict(coords, sample_weight=weights)

    sites = pd.DataFrame(km.cluster_centers_, columns=["lat", "long"]).round(6)
    local = pd.Series(weights).groupby(labels).sum().reindex(range(n_sites)).fillna(0).to_numpy()

    # busier areas get bigger stores, scaled into [CAP_MIN, CAP_MAX] (rounded to 50)
    span = max(local.max() - local.min(), 1e-9)
    cap = CAP_MIN + (local - local.min()) / span * (CAP_MAX - CAP_MIN)
    cap = (np.round(cap / 50) * 50).astype(int)

    noise = 1 + rng.uniform(-FIXED_NOISE, FIXED_NOISE, n_sites)
    fixed = ((FIXED_BASE + FIXED_PER_CAP * cap) * noise).round(-2).astype(int)

    sites.insert(0, "site_id", [f"DS{j:02d}" for j in range(1, n_sites + 1)])
    sites["local_demand"] = local.round(1)
    sites["capacity_per_day"] = cap
    sites["fixed_cost_per_day"] = fixed

    return sites


def inr(x, decimals=0):
    """Format a number in the Indian system: 1234567 -> '12,34,567'."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "-"
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    ip, _, fp = s.partition(".")
    if len(ip) > 3:
        head, tail = ip[:-3], ip[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        ip = ",".join(groups + [tail])
    return ("-" if neg else "") + ip + ("." + fp if fp else "")


def lakh(x):
    """Rupee amount in lakh: 450701 -> '4.51 lakh'."""
    return f"{x / 1e5:.2f} lakh"


def cell_km_for(max_dist_km):
    """Grid square size used to group customers: one tenth of the distance limit
    (300 m at 3 km, 800 m at 8 km), so grouping stays fine relative to the service radius."""
    return max_dist_km / 10.0


def group_customers_grid(cust, cell_km, reach=None):
    """Group customers that fall in the same square grid cell (side cell_km) into one demand point.
    If `reach` (n x m boolean: which stores each customer may use) is given, customers are grouped only
    when they can also reach exactly the same set of stores, so grouping never changes feasibility.
    Each point sits at the demand-weighted centre of its customers and carries their total demand.
    Returns (points DataFrame: id, lat, long, points, n_customers;  group label of each customer)."""
    lat = cust["lat"].to_numpy(); lon = cust["long"].to_numpy()
    w = cust["points"].to_numpy(float)
    cell_lat = cell_km / 111.0
    cell_lon = cell_km / (111.0 * np.cos(np.radians(lat.mean())))
    keys = np.stack([np.floor(lat / cell_lat), np.floor(lon / cell_lon)], axis=1)
    if reach is not None:
        keys = np.hstack([keys, np.packbits(reach, axis=1).astype(float)])
    _, labels = np.unique(keys, axis=0, return_inverse=True)
    labels = labels.ravel()
    wsum = np.bincount(labels, weights=w)
    pts = pd.DataFrame({"id": np.arange(1, len(wsum) + 1),
                        "lat": np.bincount(labels, weights=w * lat) / wsum,
                        "long": np.bincount(labels, weights=w * lon) / wsum,
                        "points": wsum, "n_customers": np.bincount(labels)})
    return pts, labels


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on NumPy arrays (broadcasting)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def distance_matrix(cust, sites):
    """Road-distance matrix (n_customers x n_sites), km."""
    return haversine_km(cust["lat"].to_numpy()[:, None], cust["long"].to_numpy()[:, None],
                        sites["lat"].to_numpy()[None, :], sites["long"].to_numpy()[None, :]) * ROAD_FACTOR


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cust = generate_customers(10000)
    sites = make_candidates(cust, 50)
    cust.to_csv("chennai_customers_10000.csv", index=False)
    sites.to_csv("chennai_sites_50.csv", index=False)

    d = distance_matrix(cust, sites)
    print("Customers:", len(cust), "| total demand (orders/day):", cust.points.sum())
    print("Candidates:", len(sites), "| d shape:", d.shape)
    print(sites[["capacity_per_day", "fixed_cost_per_day"]].describe().round(0))

    plt.figure(figsize=(6, 8))
    plt.scatter(cust["long"], cust["lat"], s=1, alpha=0.3, label="Customers")
    plt.scatter(sites["long"], sites["lat"], c="red", s=40, marker="^", label="Candidate sites")
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Chennai: customers & K-Means candidates")
    plt.legend(); plt.tight_layout(); plt.savefig("data_map.png", dpi=110); plt.show()
