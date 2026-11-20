"""Rolling-origin forecasting benchmark for weekly Ebola incidence (Guinea, Liberia, Sierra Leone).

Everything a model uses at forecast origin t is information available at t:
  * features and training targets come from a *causal* weekly series built from the raw cumulative counts
    truncated at t (negative weeks netted, one-off backlog dumps capped using only past weeks);
  * models are trained only on rows whose targets were already observed by t;
  * forecasts are scored against the retrospectively cleaned incidence (the best estimate of the truth).

Usage: python ebola_ml_benchmark.py <ebola_data_db_format.csv> <outdir>
"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

SRC, OUT = sys.argv[1], sys.argv[2]
IND = "Cumulative number of confirmed, probable and suspected Ebola cases"
COUNTRIES = ["Guinea", "Liberia", "Sierra Leone"]
POP = {"Guinea": 12.0e6, "Liberia": 4.3e6, "Sierra Leone": 6.2e6}
R0_LIT = {"Guinea": 1.71, "Liberia": 1.83, "Sierra Leone": 2.02}
SIGMA, GAMMA = 1 / 11.4, 1 / 7.5
HORIZONS = [1, 2, 4]
MIN_ORIGIN, STEP = 20, 2          # first forecast origin (weeks of history) and spacing
MIN_TRAIN_ROWS = 30
raw = pd.read_csv(SRC, parse_dates=["Date"])

# ---------------------------------------------------------------- data
def weekly_cum(country):
    s = (raw[(raw.Country == country) & (raw.Indicator == IND)]
         .drop_duplicates("Date", keep="last").set_index("Date").value.sort_index())
    return s.resample("W").last().ffill()

CUM = {c: weekly_cum(c) for c in COUNTRIES}
T = min(len(v) for v in CUM.values()) - 1                       # number of weekly increments
DATES = CUM["Guinea"].index[1:T + 1]

def causal_series(cum_values):
    """Weekly incidence from cumulative counts available so far (length n+1 -> n weeks). Past data only."""
    inc = np.diff(np.asarray(cum_values, float))
    out = []
    for i, v in enumerate(inc):
        if v < 0 and out:                                       # reclassification: net with previous week
            tot = out[-1] + v
            out[-1] = out[-1] = max(tot / 2, 0); v = max(tot / 2, 0)
        elif v < 0:
            v = 0.0
        past = out[-4:]
        if len(past) >= 3:                                      # backlog dump: cap at 1.5x median of previous weeks
            med = np.median(past)
            if v > 3 * med and v > 300:
                v = 1.5 * med
        out.append(v)
    return np.array(out)

def retro_clean(country):
    """Retrospective cleaning used as ground truth (same rules as ebola_model.py)."""
    w = CUM[country]
    inc = w.diff().dropna().values.astype(float)
    for i in range(1, len(inc)):
        if inc[i] < 0:
            tot = inc[i - 1] + inc[i]; inc[i - 1] = inc[i] = tot / 2
    for i in range(len(inc)):
        lo, hi = max(0, i - 3), min(len(inc), i + 4)
        med = np.median(np.delete(inc[lo:hi], i - lo))
        if inc[i] > 3 * med and inc[i] > 300:
            ex = inc[i] - 1.5 * med; inc[i] -= ex
            k = min(8, i)
            if k: inc[i - k:i] += ex / k
    return np.clip(inc, 0, None)

TRUTH = {c: retro_clean(c)[:T] for c in COUNTRIES}

# ---------------------------------------------------------------- features
FEATS = ["lg0", "lg1", "lg2", "lg3", "d1", "d2", "d4", "m4", "sd4"]

def features(arr, s):
    lg = np.log1p(arr)
    w = lg[s - 3:s + 1]
    return [lg[s], lg[s - 1], lg[s - 2], lg[s - 3], lg[s] - lg[s - 1], lg[s] - lg[s - 2], lg[s] - lg[s - 4], w.mean(), w.std()]

def build_rows(series_by_country, countries, h, upto):
    X, y = [], []
    for c in countries:
        a = series_by_country[c]; lg = np.log1p(a)
        for s in range(4, upto - h + 1):
            X.append(features(a, s)); y.append(lg[s + h] - lg[s])
    return np.array(X), np.array(y)

# ---------------------------------------------------------------- SEIR (R0 fixed to literature; fit rho, t0, k)
def seir_weekly(beta0, rho, t0, k, weeks, N, init, dt=1.0):
    C0, i0 = init
    E, I = i0 / SIGMA, i0 / GAMMA
    S = N - C0 - E - I
    days = weeks * 7
    out = np.zeros(days); steps = int(1 / dt)
    for d in range(days):
        n_i = 0.0
        for s_ in range(steps):
            t = d + s_ * dt
            b = beta0 * (rho + (1 - rho) / (1 + np.exp(k * (t - t0))))
            ne = b * S * I / N * dt; ni = SIGMA * E * dt; rc = GAMMA * I * dt
            S -= ne; E += ne - ni; I += ni - rc; n_i += ni
        out[d] = n_i
    return out.reshape(weeks, 7).sum(axis=1)

def seir_forecast(country, cs, c0, hmax):
    n = len(cs)
    init = (float(c0), float(cs[:2].mean() / 7))
    beta0 = R0_LIT[country] * GAMMA

    def resid(p):
        m = seir_weekly(beta0, p[0], p[1], p[2], n, POP[country], init)
        return np.sqrt(m + 1) - np.sqrt(cs + 1)

    best = None
    for rho0, t00 in ((0.3, 10), (0.6, 80)):
        r = least_squares(resid, [rho0, t00, 0.05], bounds=([0, 0, 0.005], [1.2, 400, 0.5]), max_nfev=25)
        if best is None or r.cost < best.cost: best = r
    p = best.x
    return seir_weekly(beta0, p[0], p[1], p[2], n + hmax, POP[country], init)[n:]

# ---------------------------------------------------------------- rolling-origin loop
rows = []
origins = list(range(MIN_ORIGIN, T - 1, STEP))
print(f"{len(origins)} origins x {len(COUNTRIES)} countries x horizons {HORIZONS}; T={T} weeks")
for t in origins:
    cs = {c: causal_series(CUM[c].values[:t + 2]) for c in COUNTRIES}    # weeks 0..t known at origin t
    cur = {c: cs[c][t] for c in COUNTRIES}
    seir_fc = {c: seir_forecast(c, cs[c], CUM[c].values[0], max(HORIZONS)) for c in COUNTRIES}
    for h in HORIZONS:
        if t + h > T - 1: continue
        Xall, yall = build_rows(cs, COUNTRIES, h, t)
        models = {}
        if len(yall) >= MIN_TRAIN_ROWS:
            sc = StandardScaler().fit(Xall)
            models["Ridge AR"] = (Ridge(alpha=3.0).fit(sc.transform(Xall), yall), sc)
            models["GBM"] = (GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                      loss="huber", random_state=0).fit(Xall, yall), None)
        for c in COUNTRIES:
            a = cs[c]; lg = np.log1p(a)
            truth = TRUTH[c][t + h]
            f = np.array(features(a, t))[None, :]
            pred = {"Persistence": a[t],
                    "4-wk average": a[t - 3:t + 1].mean()}
            sl = np.clip(np.polyfit(np.arange(5), lg[t - 4:t + 1], 1)[0], -0.4, 0.4)
            pred["Exp. trend"] = np.expm1(lg[t] + sl * h * 0.7)                     # damped log-linear extrapolation
            pred["SEIR (refit)"] = seir_fc[c][h - 1]
            for name, (m, sc_) in models.items():
                z = sc_.transform(f) if sc_ is not None else f
                pred[name] = np.expm1(lg[t] + m.predict(z)[0])
            if len(yall) >= MIN_TRAIN_ROWS:                                         # new-country test: train on the other two only
                others = [o for o in COUNTRIES if o != c]
                Xo, yo = build_rows(cs, others, h, t)
                if len(yo) >= 20:
                    g = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                  loss="huber", random_state=0).fit(Xo, yo)
                    pred["GBM (new country)"] = np.expm1(lg[t] + g.predict(f)[0])
            for name, v in pred.items():
                rows.append(dict(origin=t, date=DATES[t], country=c, h=h, model=name, pred=max(float(v), 0.0), truth=float(truth),
                                 current=float(cur[c])))
    print(f"  origin week {t} ({DATES[t].date()}) done")

df = pd.DataFrame(rows)
df["abs_err"] = (df.pred - df.truth).abs()
df["log_err"] = (np.log1p(df.pred) - np.log1p(df.truth)).abs()
df["phase"] = np.where(df.date <= pd.Timestamp("2015-03-01"), "Active phase (to Mar 2015)", "Decline & tail (after Mar 2015)")
df.to_csv(f"{OUT}/benchmark_predictions.csv", index=False)

# ---------------------------------------------------------------- scoring
# compare all models on the same (origin, country, h) rows that every model has
full = df.groupby(["origin", "country", "h"]).model.transform("nunique") == df.model.nunique()
d = df[full].copy()
print("Rows scored per model:", len(d) // d.model.nunique())

def summarise(sub):
    g = sub.groupby("model")[["abs_err", "log_err"]].mean()
    base = g.loc["Persistence"]
    return pd.DataFrame({"MAE": g.abs_err, "MAE_vs_persist": g.abs_err / base.abs_err,
                         "logMAE": g.log_err, "logMAE_vs_persist": g.log_err / base.log_err})

tables = {}
for ph in [None] + list(d.phase.unique()):
    for h in HORIZONS:
        sub = d[(d.h == h) & ((d.phase == ph) if ph else True)]
        tables[(ph or "All", h)] = summarise(sub)
for k, v in tables.items():
    if k[0] == "All":
        print(f"\n=== {k[0]}, h={k[1]} week(s) ahead ==="); print(v.round(3).sort_values("logMAE").to_string())

# block bootstrap: is the best ML model reliably better than persistence?
def boot_ratio(sub, model, n=2000, block=2, seed=0):
    r = np.random.default_rng(seed)
    p = sub[sub.model == model].set_index(["origin", "country"]).log_err
    b = sub[sub.model == "Persistence"].set_index(["origin", "country"]).log_err
    diff = (p - b).groupby(level=0).mean()                         # per-origin mean difference (negative = better than persistence)
    v = diff.values; m = len(v); out = []
    for _ in range(n):
        starts = r.integers(0, m, size=int(np.ceil(m / block)))
        idx = np.concatenate([np.arange(s, s + block) % m for s in starts])[:m]
        out.append(v[idx].mean())
    return float(v.mean()), float(np.percentile(out, 5)), float(np.percentile(out, 95))

boot = {}
for h in HORIZONS:
    for mdl in ["GBM", "Ridge AR", "GBM (new country)", "SEIR (refit)", "Exp. trend"]:
        boot[f"{mdl}|h{h}"] = boot_ratio(d[d.h == h], mdl)
        mean, lo, hi = boot[f"{mdl}|h{h}"]
        print(f"h={h} {mdl:18s} mean log-error diff vs persistence {mean:+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]")

# feature importance (final GBM, 2 weeks ahead)
cs_full = {c: causal_series(CUM[c].values) for c in COUNTRIES}
Xf, yf = build_rows(cs_full, COUNTRIES, 2, len(cs_full["Guinea"]) - 1)
imp = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, loss="huber", random_state=0).fit(Xf, yf)
importance = dict(zip(FEATS, map(float, imp.feature_importances_)))
print("\nGBM feature importance (h=2, full data):", {k: round(v, 3) for k, v in sorted(importance.items(), key=lambda x: -x[1])})

# ---------------------------------------------------------------- charts
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
order = ["Persistence", "4-wk average", "Exp. trend", "SEIR (refit)", "Ridge AR", "GBM", "GBM (new country)"]
colors = {"Persistence": "#999999", "4-wk average": "#bbbbbb", "Exp. trend": "#e9c46a", "SEIR (refit)": "#264653",
          "Ridge AR": "#f4a261", "GBM": "#e76f51", "GBM (new country)": "#b5503a"}
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
for a, ph in zip(ax, sorted(d.phase.unique())):
    w = 0.12
    for i, mdl in enumerate(order):
        vals = [tables[(ph, h)].loc[mdl, "logMAE_vs_persist"] for h in HORIZONS]
        a.bar(np.arange(len(HORIZONS)) + (i - 3) * w, vals, w, color=colors[mdl], label=mdl)
    a.axhline(1, color="k", lw=1, ls="--")
    a.set_xticks(range(len(HORIZONS))); a.set_xticklabels([f"{h} wk ahead" for h in HORIZONS])
    a.set_title(ph)
ax[0].set_ylabel("Forecast error relative to persistence\n(log-scale MAE; below 1 = better)")
ax[0].legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_ml_benchmark.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, c in zip(ax, COUNTRIES):
    s = df[(df.country == c) & (df.h == 2)]
    tr = TRUTH[c]
    a.plot(DATES, tr, color="k", lw=1.5, label="Truth (cleaned)")
    for mdl in ["Persistence", "SEIR (refit)", "GBM"]:
        m = s[s.model == mdl]
        a.plot(m.date + pd.Timedelta(weeks=2), m.pred, "o-", ms=3, lw=1, color=colors[mdl], label=mdl)
    a.set_title(f"{c}: 2-week-ahead forecasts"); a.tick_params(axis="x", rotation=30)
ax[0].set_ylabel("Cases per week"); ax[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_forecast_examples.png", dpi=150); plt.close()

# ---------------------------------------------------------------- save
res = {"setup": dict(origins=len(origins), step_weeks=STEP, first_origin_week=MIN_ORIGIN, horizons=HORIZONS, weeks=T),
       "tables": {f"{k[0]}|h{k[1]}": v.round(4).to_dict("index") for k, v in tables.items()},
       "bootstrap_logerr_diff_vs_persistence": {k: dict(mean=v[0], p5=v[1], p95=v[2]) for k, v in boot.items()},
       "gbm_feature_importance": importance}
json.dump(res, open(f"{OUT}/summary_ml.json", "w"), indent=2)
