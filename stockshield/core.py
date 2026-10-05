"""StockShield core: ingest -> validate -> censored demand -> features -> forecast -> risk -> alerts -> SQLite."""
import json, math, sqlite3, sys, os, hashlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, roc_auc_score

MODEL_VERSION = "rf-v2-catalogue"
THRESH = {"HIGH": 0.6, "MEDIUM": 0.3}  # risk_probability cut-offs
FEATS = ["mid", "lag0", "lag6", "lag13", "m7", "m28", "sd7", "trend", "accel", "dow", "month", "inv", "lead", "safety", "so30"]
TABLES = ["medicine", "inventory", "sales", "stockout_event", "forecast", "risk_prediction", "alert", "evaluation", "medicine_catalogue"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS medicine(id INTEGER PRIMARY KEY,name TEXT,category TEXT,unit TEXT,supplier TEXT,lead_time_days INTEGER,safety_stock INTEGER);
CREATE TABLE IF NOT EXISTS medicine_catalogue(catalogue_id TEXT PRIMARY KEY,name TEXT,category TEXT,dosage_form TEXT,strength TEXT,manufacturer TEXT,indication TEXT,classification TEXT,source TEXT);
CREATE INDEX IF NOT EXISTS idx_catalogue_name ON medicine_catalogue(name);
CREATE INDEX IF NOT EXISTS idx_catalogue_category ON medicine_catalogue(category);
CREATE TABLE IF NOT EXISTS inventory(medicine_id INTEGER,date TEXT,quantity_available REAL,quantity_received REAL,PRIMARY KEY(medicine_id,date));
CREATE TABLE IF NOT EXISTS sales(medicine_id INTEGER,date TEXT,quantity_sold REAL,PRIMARY KEY(medicine_id,date));
CREATE TABLE IF NOT EXISTS stockout_event(medicine_id INTEGER,start_date TEXT,end_date TEXT,duration INTEGER);
CREATE TABLE IF NOT EXISTS forecast(medicine_id INTEGER,forecast_date TEXT,predicted_demand REAL,lower_bound REAL,upper_bound REAL,model_version TEXT);
CREATE TABLE IF NOT EXISTS risk_prediction(medicine_id INTEGER,prediction_date TEXT,risk_probability REAL,risk_level TEXT,expected_stockout_date TEXT,days_remaining REAL,explanation TEXT);
CREATE TABLE IF NOT EXISTS alert(medicine_id INTEGER,type TEXT,severity TEXT,message TEXT);
CREATE TABLE IF NOT EXISTS evaluation(key TEXT PRIMARY KEY,value TEXT);
"""
# name, category, supplier, lead, safety, base/day, trend/yr, season amp, noise
MEDS = [("Paracetamol 500mg", "Analgesic", "MedSupply", 3, 60, 50, .10, .10, .15), ("Amoxicillin 250mg", "Antibiotic", "PharmaDist", 6, 40, 30, .05, .25, .25),
        ("Cetirizine 10mg", "Antihistamine", "MedSupply", 4, 30, 25, .00, .45, .20), ("Insulin Glargine", "Chronic", "BioLink", 7, 20, 15, .12, .05, .10),
        ("Salbutamol Inhaler", "Respiratory", "PharmaDist", 5, 25, 18, .08, .35, .30), ("Oseltamivir 75mg", "Antiviral", "BioLink", 8, 15, 10, .00, .60, .45)]


def synth(seed=7, days=420, start="2025-01-01"):
    """Synthetic pharmacy data: weekly+annual seasonality, trend, noise, spikes, reorder policy with random delays (-> stock-outs)."""
    rng = np.random.default_rng(seed); dates = pd.date_range(start, periods=days); meds, sales, inv = [], [], []
    dow, doy = dates.dayofweek.to_numpy(), dates.dayofyear.to_numpy()
    for i, (n, c, s, lead, safety, base, tr, amp, vol) in enumerate(MEDS, 1):
        meds.append((i, n, c, "unit", s, lead, safety)); t = np.arange(days)
        lam = base * (1 + tr * t / 365) * (1 + amp * np.sin(2 * np.pi * (doy - 30) / 365)) * (1 + .2 * np.isin(dow, [0, 1]) - .3 * (dow == 6))
        for _ in range(2):
            a = rng.integers(40, days - 10); lam[a:a + 5] *= 2.2
        demand = rng.poisson(lam * np.clip(rng.normal(1, vol / 2, days), .3, None)); stock, pipe = base * lead * 2.0, []
        for k in range(days):
            rec = float(sum(q for d, q in pipe if d == k)); pipe = [(d, q) for d, q in pipe if d != k]
            stock += rec; sold = float(min(demand[k], stock)); stock -= sold
            if stock + sum(q for _, q in pipe) <= base * lead * 1.3: pipe.append((k + lead + int(rng.integers(0, 4)), base * lead * 3.0))
            sales.append((i, dates[k].strftime("%Y-%m-%d"), sold)); inv.append((i, dates[k].strftime("%Y-%m-%d"), float(stock), rec))
    return (pd.DataFrame(meds, columns=["id", "name", "category", "unit", "supplier", "lead_time_days", "safety_stock"]),
            pd.DataFrame(sales, columns=["medicine_id", "date", "quantity_sold"]), pd.DataFrame(inv, columns=["medicine_id", "date", "quantity_available", "quantity_received"]))


def ingest_catalogue(con, path=None):
    """Import the supplied 50k-row medicine metadata catalogue with stable content-derived IDs."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medicine_dataset.csv")
    if not os.path.exists(path):
        return {"loaded": False, "reason": "catalogue file not found", "records": 0}
    raw = pd.read_csv(path, dtype=str).fillna("")
    required = ["Name", "Category", "Dosage Form", "Strength", "Manufacturer", "Indication", "Classification"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"catalogue missing columns: {missing}")
    raw = raw[required].copy()
    for c in required: raw[c] = raw[c].astype(str).str.strip()
    raw = raw[(raw["Name"] != "")].drop_duplicates().reset_index(drop=True)
    rows=[]
    for r in raw.itertuples(index=False):
        vals=list(r); key="|".join(vals).lower().encode(); cid=hashlib.sha256(key).hexdigest()[:24]
        rows.append((cid,*vals,"medicine_dataset.csv"))
    con.executemany("INSERT INTO medicine_catalogue VALUES(?,?,?,?,?,?,?,?,?)", rows)
    return {"loaded": True, "records": len(rows), "unique_names": int(raw["Name"].nunique()), "source": "medicine_dataset.csv", "columns": required}


def validate(meds, sales, inv):
    for name, df in (("sales", sales), ("inventory", inv)):
        if df.duplicated(["medicine_id", "date"]).any(): raise ValueError(f"{name}: duplicate (medicine_id,date) rows")
        if not set(df.medicine_id) <= set(meds.id): raise ValueError(f"{name}: unknown medicine_id")
    if (sales.quantity_sold < 0).any() or (inv.quantity_available < 0).any(): raise ValueError("negative quantities")


def process(meds, sales, inv):
    """Censored-demand treatment. End-of-day stock == 0 => that day's sales are a LOWER BOUND of demand (censored).
    Latent demand = trailing 28-day mean of uncensored sales (past data only -> no leakage); demand_adj = max(sales, latent) on censored days.
    Raw quantity_sold is preserved for auditability."""
    m = meds.rename(columns={"id": "medicine_id"})[["medicine_id", "lead_time_days", "safety_stock"]]
    df = sales.merge(inv, on=["medicine_id", "date"]).merge(m, on="medicine_id"); df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["medicine_id", "date"]).reset_index(drop=True); df["censored"] = df.quantity_available <= 0; out = []
    for _, g in df.groupby("medicine_id"):
        g = g.copy(); latent = g.quantity_sold.where(~g.censored).rolling(28, min_periods=3).mean().shift(1).ffill()
        g["demand_adj"] = np.where(g.censored, np.maximum(g.quantity_sold, latent.fillna(g.quantity_sold)), g.quantity_sold); out.append(g)
    return pd.concat(out, ignore_index=True)


def detect_stockouts(df):
    rows = []
    for mid, g in df.groupby("medicine_id"):
        run = (g.censored != g.censored.shift()).cumsum()
        for _, r in g[g.censored].groupby(run[g.censored]):
            rows.append((int(mid), r.date.min().strftime("%Y-%m-%d"), r.date.max().strftime("%Y-%m-%d"), len(r)))
    return pd.DataFrame(rows, columns=["medicine_id", "start_date", "end_date", "duration"])


def features(df, h):
    """Row at date t uses only data up to t. Targets: y = demand over t+1..t+h; yL = demand over the supplier lead window."""
    out = []
    for mid, g in df.groupby("medicine_id"):
        g = g.sort_values("date"); d = g.demand_adj.reset_index(drop=True); L = int(g.lead_time_days.iloc[0]); m7 = d.rolling(7).mean(); m28 = d.rolling(28).mean()
        out.append(pd.DataFrame({"medicine_id": mid, "date": g.date.values, "mid": mid, "lag0": d.values, "lag6": d.shift(6).values, "lag13": d.shift(13).values,
                                 "m7": m7.values, "m28": m28.values, "sd7": d.rolling(7).std().values, "trend": (m7 / m28.replace(0, np.nan)).values - 1,
                                 "accel": (m7 - m7.shift(7)).values, "dow": g.date.dt.dayofweek.values, "month": g.date.dt.month.values, "inv": g.quantity_available.values,
                                 "lead": g.lead_time_days.values, "safety": g.safety_stock.values, "so30": g.censored.astype(int).rolling(30, min_periods=1).sum().values,
                                 "y": d.rolling(h).sum().shift(-h).values, "yL": d.rolling(L).sum().shift(-L).values}))
    return pd.concat(out, ignore_index=True).dropna(subset=FEATS)


def train(df, h):
    f = features(df, h); lab = f.dropna(subset=["y"]); cut = lab.date.quantile(.8)
    tr, va = lab[lab.date <= cut - pd.Timedelta(days=h)], lab[lab.date > cut]  # temporal split, gap = horizon
    mk = lambda: RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=-1)
    p = mk().fit(tr[FEATS], tr.y).predict(va[FEATS]); base = va.m7 * h; rmse = lambda a, b: float(math.sqrt(mean_squared_error(a, b)))
    res = dict(horizon=h, n_train=len(tr), n_val=len(va), rf_mae=float(mean_absolute_error(va.y, p)), rf_rmse=rmse(va.y, p),
               moving_avg_mae=float(mean_absolute_error(va.y, base)), moving_avg_rmse=rmse(va.y, base), sigma=float((va.y - p).std()))
    va = va.assign(pred=p, base=base); res["by_medicine"] = {str(k): dict(rf_mae=float(mean_absolute_error(g.y, g.pred)), moving_avg_mae=float(mean_absolute_error(g.y, g.base)), n=len(g)) for k, g in va.groupby("medicine_id")}
    return mk().fit(lab[FEATS], lab.y), res, va


def phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def level(p): return "HIGH" if p >= THRESH["HIGH"] else "MEDIUM" if p >= THRESH["MEDIUM"] else "LOW"


def shortfall_prob(stock, daily, lead, safety, sigma7):
    """P(demand during supplier lead time > stock - safety_stock); normal approx, sd scaled from validation residuals."""
    return 1 - phi((stock - safety - daily * lead) / max(sigma7 * math.sqrt(lead / 7), 1e-6))


def shortage(stock, fc):
    """Return stock coverage in days and the first forecast day that reaches zero.

    This calculation is deliberately independent of catalogue size or medicine count.
    It uses the actual stock quantity and per-day demand. For horizons shorter than the
    projected coverage, demand is extrapolated from the positive forecast rate rather
    than clamping the answer to the forecast horizon. Large inventories therefore produce
    proportionally larger coverage (e.g. 56 vs 56,000 units cannot both return 0.5 days).
    """
    stock = max(float(stock or 0.0), 0.0)
    fc = np.asarray(fc, dtype=float)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    if stock <= 0:
        return 0.0, 1 if len(fc) else None
    if len(fc) == 0:
        return float("inf"), None
    fc = np.maximum(fc, 0.0)
    cum = np.cumsum(fc)
    hit = np.flatnonzero(cum >= stock)
    if hit.size:
        k = int(hit[0])
        before = float(cum[k - 1]) if k else 0.0
        day_demand = max(float(fc[k]), 1e-9)
        return float(k + (stock - before) / day_demand), k + 1
    recent = fc[-7:] if len(fc) >= 7 else fc
    rate = float(np.mean(recent[recent > 0])) if np.any(recent > 0) else 0.0
    if rate <= 0:
        return float("inf"), None
    return float(len(fc) + max(stock - float(cum[-1]), 0.0) / rate), None


def coverage_days(stock, daily_demand):
    """Direct scalar stock coverage: stock units / positive daily demand."""
    stock = max(float(stock or 0.0), 0.0)
    daily = max(float(daily_demand or 0.0), 0.0)
    return float("inf") if daily <= 0 and stock > 0 else (0.0 if stock <= 0 else stock / daily)


def assess(row, fc, sigma7, last):
    stock, lead, safety = float(row.inv), int(row.lead), float(row.safety); days, k = shortage(stock, fc); p = shortfall_prob(stock, fc[:7].mean(), lead, safety, sigma7)
    date = (last + pd.Timedelta(days=int(math.ceil(days)))).strftime("%Y-%m-%d") if k else None; lv = level(p); f7 = float(np.sum(fc[:7]))
    why = [f"Current inventory: {stock:.0f} units (safety stock {safety:.0f})", f"Forecast demand next 7 days: {f7:.0f} units", f"Recent demand trend: {row.trend * 100:+.0f}%",
           f"Supplier lead time: {lead} days", f"Projected shortage in {days:.1f} days" if k else f"No shortage projected within 14 days (~{days:.0f} days of cover)"]
    if row.so30 > 0: why.append(f"Recorded sales were suppressed during {int(row.so30)} stock-out day(s) in the last 30 days; demand estimation was adjusted accordingly.")
    al = []
    if lv == "HIGH": al.append(("HIGH_STOCKOUT_RISK", "critical", f"Stock-out risk {p:.0%} within the {lead}-day lead time"))
    if days < lead: al.append(("SHORTAGE_BEFORE_REPLENISHMENT", "critical", f"Projected inventory runs out in {days:.1f} days; supplier lead time is {lead} days"))
    if stock < safety: al.append(("LOW_INVENTORY", "warning", f"Inventory {stock:.0f} is below safety stock {safety:.0f}"))
    if row.m28 > 0 and row.m7 > 1.4 * row.m28: al.append(("DEMAND_SPIKE", "warning", f"7-day demand is {row.m7 / row.m28 - 1:+.0%} vs 28-day average"))
    if lead >= 7 and days < lead + 3: al.append(("SUPPLIER_LEAD_TIME_RISK", "warning", f"Long lead time ({lead}d) leaves only {days - lead:.1f} days of buffer"))
    return dict(p=p, level=lv, date=date, days=days, explanation=" | ".join(why), alerts=al)


def run(db="stockshield.db", seed=7):
    meds, sales, inv = synth(seed); validate(meds, sales, inv); con = sqlite3.connect(db); con.executescript(SCHEMA)
    for t in TABLES: con.execute(f"DELETE FROM {t}")
    meds.to_sql("medicine", con, if_exists="append", index=False); sales.to_sql("sales", con, if_exists="append", index=False); inv.to_sql("inventory", con, if_exists="append", index=False)
    catalogue_meta = ingest_catalogue(con)
    df = process(*(pd.read_sql(f"SELECT * FROM {t}", con) for t in ("medicine", "sales", "inventory")))  # pipeline reads from the DB
    so = detect_stockouts(df); so.to_sql("stockout_event", con, if_exists="append", index=False); last = df.date.max()
    (m7, r7, va7), (m14, r14, _) = train(df, 7), train(df, 14); latest = features(df, 7).sort_values("date").groupby("medicine_id").tail(1)
    p7, p14 = m7.predict(latest[FEATS]), m14.predict(latest[FEATS]); sd = r7["sigma"] / math.sqrt(7); fr, rr, al = [], [], []
    for (_, row), a, b in zip(latest.iterrows(), p7, p14):
        fc = np.clip(np.r_[np.full(7, a / 7), np.full(7, max(b - a, 0) / 7)], 0, None); mid = int(row.medicine_id)
        for k, v in enumerate(fc, 1): fr.append((mid, (last + pd.Timedelta(days=k)).strftime("%Y-%m-%d"), float(v), float(max(v - 1.64 * sd, 0)), float(v + 1.64 * sd), MODEL_VERSION))
        r = assess(row, fc, r7["sigma"], last); rr.append((mid, last.strftime("%Y-%m-%d"), r["p"], r["level"], r["date"], r["days"], r["explanation"])); al += [(mid, *x) for x in r["alerts"]]
    for t, rows in (("forecast", fr), ("risk_prediction", rr), ("alert", al)):
        if rows: con.executemany(f"INSERT INTO {t} VALUES({','.join('?' * len(rows[0]))})", rows)
    v = va7.dropna(subset=["yL"]).copy(); v["p"] = [shortfall_prob(r.inv, r.pred / 7, r.lead, r.safety, r7["sigma"]) for r in v.itertuples()]; v["lab"] = (v.yL > v.inv - v.safety).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(v.lab, v.p >= .5, average="binary", zero_division=0)
    risk = dict(n=len(v), positives=int(v.lab.sum()), precision=float(pr), recall=float(rc), f1=float(f1), threshold=.5,
                roc_auc=float(roc_auc_score(v.lab, v.p)) if v.lab.nunique() == 2 else None, note=None if v.lab.nunique() == 2 else "ROC-AUC undefined: validation labels contain one class")
    con.execute("INSERT INTO evaluation VALUES('forecast',?)", (json.dumps([r7, r14]),)); con.execute("INSERT INTO evaluation VALUES('risk',?)", (json.dumps(risk),))
    share = df.groupby("medicine_id").censored.mean(); names = dict(zip(meds.id, meds.name))
    meta = dict(ran_at=pd.Timestamp.now().isoformat(timespec="seconds"), medicines=len(meds), records=len(sales), stockout_events=len(so), censored_days=int(df.censored.sum()), catalogue_records=catalogue_meta.get("records",0), catalogue_source=catalogue_meta.get("source"),
                warnings=[f"{names[i]}: {x:.0%} of days censored by stock-outs" for i, x in share.items() if x > .1])
    con.execute("INSERT INTO evaluation VALUES('meta',?)", (json.dumps(meta),)); con.commit(); con.close()
    return dict(medicines=len(meds), stockout_events=len(so), censored_days=int(df.censored.sum()), forecasts=len(fr), risks=len(rr), alerts=len(al), catalogue_records=catalogue_meta.get("records",0), forecast_eval=[r7, r14], risk_eval=risk)


if __name__ == "__main__":
    print(json.dumps(run(*(sys.argv[1:2] or ["stockshield.db"])), indent=1))
