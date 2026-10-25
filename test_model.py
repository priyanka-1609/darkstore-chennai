"""
Small (toy) test cases - solvable by hand (class slide on test-case types).
Each case checks one constraint. Run:  python test_model.py

Data: 5 customers, 2 candidate sites.
  Profit per order = rev - cpk * distance = 100 - 3 * d_ij
"""
from model import build_and_solve

D = [5, 4, 3, 2, 10]                          # orders/day
F = [200, 200]                                # Rs/day
C = [100, 100]                                # orders/day
d = [[1, 1], [2, 3], [4, 5], [12, 10], [20, 20]]   # km
rev, cpk = 100, 3


def run(name, expected, **kw):
    args = dict(D=D, F=F, C=C, d=d, rev=rev, cpk=cpk, max_dist=15, max_stores=5)
    args.update(kw)
    r = build_and_solve(**args)
    ok = abs(r["objective"] - expected) < 1e-6
    print(f"{'PASS' if ok else 'FAIL'}  {name:<55} profit={r['objective']:8.1f}  expected={expected:8.1f}  open={[int(v > 0.5) for v in r['y']]}")
    return ok


cases = [
    # Base: only site 0 opens; customers 0-3 served; customer 4 (20 km) beyond Dmax
    #   site0: 5*97 + 4*94 + 3*88 + 2*64 = 1253 - 200 = 1053
    #   site1: 5*97 + 4*91 + 3*85 + 2*70 = 1244 - 200 = 1044 ; both: 1265 - 400 = 865
    ("Base case (Dmax=15)", 1053, {}),
    # Max distance: Dmax=25 lets customer 4 be served from site 0: +10*(100-60)=400
    ("Max distance relaxed (Dmax=25)", 1453, dict(max_dist=25)),
    # Max distance tight: Dmax=1.5 -> only customer 0 reachable: 5*97 - 200 = 285
    ("Max distance tight (Dmax=1.5)", 285, dict(max_dist=1.5)),
    # Capacity: 5 orders per store -> best 10 orders: 485 (c0@s1) + 376 + 88 (c1,c2@s0) - 400
    ("Capacity binding (C=[5,5], Dmax=25)", 549, dict(C=[5, 5], max_dist=25)),
    # Fixed cost too high -> open nothing
    ("Fixed cost too high (F=[2000,2000])", 0, dict(F=[2000, 2000])),
    # Store limit: free stores -> both open (1265); with K=1 only one (1253)
    ("Store limit not binding (F=0, K=2)", 1265, dict(F=[0, 0], max_stores=2)),
    ("Store limit binding (F=0, K=1)", 1253, dict(F=[0, 0], max_stores=1)),
    ("Store limit zero (K=0)", 0, dict(max_stores=0)),
]

if __name__ == "__main__":
    results = [run(n, e, **kw) for n, e, kw in cases]
    print(f"\n{sum(results)}/{len(results)} tests passed")
    # Write the base toy model to an .lp file for inspection (as asked in class)
    build_and_solve(D, F, C, d, rev, cpk, 15, 5, lp_file="toy_model.lp")
    print("Toy model written to toy_model.lp")
