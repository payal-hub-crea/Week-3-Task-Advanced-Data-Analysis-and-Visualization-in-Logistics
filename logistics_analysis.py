"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Simulates 6,000 shipments, validates/cleans them, runs EDA and builds figures.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.0)
FIG = "figs/"
OUT = {}

# ----------------------------------------------------------------------
# 1. DATA SIMULATION
# ----------------------------------------------------------------------
N = 6000
hubs = {"Mumbai": 0.28, "Delhi": 0.24, "Chennai": 0.18, "Kolkata": 0.15, "Pune": 0.15}
regions = {"North": 0.22, "South": 0.20, "East": 0.16, "West": 0.24, "Central": 0.18}
carriers = {"SwiftLine": 0.30, "RoadKing": 0.28, "CargoPlus": 0.24, "EcoFreight": 0.18}
modes = {"Road": 0.68, "Rail": 0.22, "Air": 0.10}

# hub -> region approximate distance matrix (km)
dist_base = pd.DataFrame(
    [[1400, 1000, 2000, 350, 800],    # Mumbai
     [300, 2200, 1500, 1100, 900],    # Delhi
     [2200, 350, 1650, 1300, 1100],   # Chennai
     [1500, 1700, 400, 2000, 1100],   # Kolkata
     [1450, 850, 1800, 200, 650]],    # Pune
    index=list(hubs), columns=["North", "South", "East", "West", "Central"],
)

df = pd.DataFrame({
    "shipment_id": [f"SH{100000+i}" for i in range(N)],
    "origin_hub": rng.choice(list(hubs), N, p=list(hubs.values())),
    "dest_region": rng.choice(list(regions), N, p=list(regions.values())),
    "carrier": rng.choice(list(carriers), N, p=list(carriers.values())),
    "mode": rng.choice(list(modes), N, p=list(modes.values())),
    "priority": rng.choice(["Standard", "Express"], N, p=[0.82, 0.18]),
})

# ship date: whole of 2025, with a Q4 peak (Oct-Dec weighted heavier)
month_w = np.array([0.070, 0.065, 0.075, 0.075, 0.080, 0.075,
                    0.080, 0.085, 0.090, 0.100, 0.115, 0.090])
month_w /= month_w.sum()
months = rng.choice(np.arange(1, 13), N, p=month_w)
days = rng.integers(1, 28, N)
df["ship_date"] = pd.to_datetime(dict(year=2025, month=months, day=days))

df["distance_km"] = [
    dist_base.loc[h, r] * rng.normal(1.0, 0.06)
    for h, r in zip(df.origin_hub, df.dest_region)
]
df["distance_km"] = df["distance_km"].round(0)

df["weight_kg"] = np.clip(rng.lognormal(mean=5.6, sigma=0.9, size=N), 20, 9000).round(1)
df["volume_m3"] = (df.weight_kg / rng.normal(180, 35, N).clip(80, 400)).round(2)
df["units"] = np.maximum(1, (df.weight_kg / rng.normal(12, 3, N).clip(4, 30)).round()).astype(int)

# ---- transport cost model (INR): mode rate per tonne-km + fixed + noise
rate = {"Road": 4.2, "Rail": 2.6, "Air": 16.0}       # INR per tonne-km
fixed = {"Road": 900, "Rail": 1500, "Air": 3500}
carrier_mult = {"SwiftLine": 1.00, "RoadKing": 0.95, "CargoPlus": 1.08, "EcoFreight": 0.92}
tonnes = df.weight_kg / 1000
df["transport_cost"] = (
    df["mode"].map(fixed)
    + df["mode"].map(rate) * df.distance_km * np.maximum(tonnes, 0.25)
)
df["transport_cost"] *= df.carrier.map(carrier_mult)
df.loc[df.priority == "Express", "transport_cost"] *= 1.25
df["transport_cost"] *= rng.normal(1.0, 0.07, N)
df["transport_cost"] = df.transport_cost.round(0)

# ---- delivery time model
speed = {"Road": 520, "Rail": 380, "Air": 1800}      # km per transit-day equivalent
df["planned_days"] = np.ceil(df.distance_km / df["mode"].map(speed) + 1).astype(int)
df.loc[df.priority == "Express", "planned_days"] = np.maximum(
    1, df.planned_days - 1
)

delay_mu = np.full(N, 0.30)
delay_mu += df.carrier.map({"SwiftLine": 0.0, "RoadKing": 0.05, "CargoPlus": 0.85, "EcoFreight": 0.12})
delay_mu += df["mode"].map({"Road": 0.10, "Rail": 0.30, "Air": -0.10})
delay_mu += np.where(df.ship_date.dt.month.isin([10, 11, 12]), 0.55, 0.0)  # peak season
delay_mu += np.where(df.origin_hub == "Kolkata", 0.35, 0.0)                # congested hub
delay_mu += np.where((df.origin_hub == "Kolkata") & (df.dest_region == "North"), 0.60, 0.0)
delay_mu += np.where(df.ship_date.dt.month.isin([7, 8, 9]), 0.20, 0.0)     # monsoon
delay_mu += df.distance_km / 6000
delay_mu += np.where(df.priority == "Express", -0.15, 0.0)
delay_mu = np.clip(delay_mu, 0.05, None)
df["delay_days"] = (rng.gamma(shape=1.2, scale=delay_mu / 1.2) - 0.65).round(0)
df["actual_days"] = np.maximum(1, df.planned_days + df.delay_days).astype(int)
df["delay_days"] = (df.actual_days - df.planned_days)

raw = df.copy()

# ---- inject realistic data quality problems (for the validation step)
inj = rng.choice(N, 150, replace=False)
df.loc[inj[:60], "transport_cost"] = np.nan
df.loc[inj[60:105], "weight_kg"] = np.nan
df.loc[inj[105:130], "carrier"] = np.nan
df.loc[inj[130:], "distance_km"] = np.nan
neg = rng.choice(N, 12, replace=False)
df.loc[neg, "weight_kg"] = -df.loc[neg, "weight_kg"]              # sign errors
ext = rng.choice(N, 8, replace=False)
df.loc[ext, "transport_cost"] = df.loc[ext, "transport_cost"] * 40  # keying errors
dups = df.sample(45, random_state=1)
df = pd.concat([df, dups], ignore_index=True)

OUT["rows_raw"] = int(len(df))

# ----------------------------------------------------------------------
# 2. DATA VALIDATION & CLEANING
# ----------------------------------------------------------------------
checks = {}
checks["duplicate_shipment_ids"] = int(df.duplicated("shipment_id").sum())
checks["missing_transport_cost"] = int(df.transport_cost.isna().sum())
checks["missing_weight_kg"] = int(df.weight_kg.isna().sum())
checks["missing_carrier"] = int(df.carrier.isna().sum())
checks["missing_distance_km"] = int(df.distance_km.isna().sum())
checks["negative_weight"] = int((df.weight_kg < 0).sum())
checks["actual_lt_1_day"] = int((df.actual_days < 1).sum())
checks["ship_date_outside_2025"] = int((df.ship_date.dt.year != 2025).sum())
# cost-per-tonne-km sanity range check (computed before outlier removal)
df["cost_per_tkm"] = df.transport_cost / (np.maximum(df.weight_kg.abs(), 250) / 1000 * df.distance_km)
def outlier_mask(d):
    """3xIQR fence on log(cost per tonne-km), computed separately per transport mode."""
    lg = np.log(d.cost_per_tkm)
    mask = pd.Series(False, index=d.index)
    for _, idx in d.groupby("mode").groups.items():
        q1, q3 = lg.loc[idx].quantile([0.25, 0.75])
        i = q3 - q1
        mask.loc[idx] = (lg.loc[idx] > q3 + 3 * i) | (lg.loc[idx] < q1 - 3 * i)
    return mask
_tmp = df.dropna(subset=["transport_cost", "weight_kg", "distance_km"]).copy()
_tmp["weight_kg"] = _tmp.weight_kg.abs()
_tmp["cost_per_tkm"] = _tmp.transport_cost / (np.maximum(_tmp.weight_kg, 250) / 1000 * _tmp.distance_km)
checks["extreme_cost_outliers_3xIQR"] = int(outlier_mask(_tmp).sum())
OUT["checks_before"] = checks

clean = df.drop_duplicates("shipment_id").copy()
clean["weight_kg"] = clean.weight_kg.abs()                       # fix sign errors
clean = clean.dropna(subset=["transport_cost", "weight_kg", "carrier", "distance_km"])
clean["cost_per_tkm"] = clean.transport_cost / (np.maximum(clean.weight_kg, 250) / 1000 * clean.distance_km)
om = outlier_mask(clean)
n_out = int(om.sum())
clean = clean[~om].copy()
OUT["rows_clean"] = int(len(clean))
OUT["removed"] = {
    "duplicates": checks["duplicate_shipment_ids"],
    "missing_any_critical_field": OUT["rows_raw"] - checks["duplicate_shipment_ids"] - len(clean) - n_out,
    "extreme_outliers": n_out,
}

# post-cleaning checks
post = {
    "duplicate_shipment_ids": int(clean.duplicated("shipment_id").sum()),
    "missing_values_total": int(clean[["transport_cost", "weight_kg", "carrier", "distance_km"]].isna().sum().sum()),
    "negative_weight": int((clean.weight_kg < 0).sum()),
    "actual_lt_1_day": int((clean.actual_days < 1).sum()),
    "cost_per_tkm_max": round(float(clean.cost_per_tkm.max()), 2),
}
OUT["checks_after"] = post
OUT["retention_pct"] = round(100 * len(clean) / (OUT["rows_raw"] - checks["duplicate_shipment_ids"]), 2)

# derived features
clean["month"] = clean.ship_date.dt.month
clean["on_time"] = (clean.delay_days <= 0).astype(int)
clean["cost_per_kg"] = clean.transport_cost / clean.weight_kg
clean["cost_per_unit"] = clean.transport_cost / clean.units
clean["quarter"] = "Q" + clean.ship_date.dt.quarter.astype(str)
clean["lane"] = clean.origin_hub + " -> " + clean.dest_region
clean.to_csv("logistics_clean.csv", index=False)
clean.head(25).to_csv("logistics_sample.csv", index=False)

# ----------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------
num_cols = ["distance_km", "weight_kg", "volume_m3", "units",
            "transport_cost", "planned_days", "actual_days", "delay_days"]
desc = clean[num_cols].describe().T
desc["median"] = clean[num_cols].median()
desc["skew"] = clean[num_cols].skew()
desc["cv_pct"] = 100 * clean[num_cols].std() / clean[num_cols].mean()
OUT["describe"] = desc.round(2).to_dict("index")
OUT["on_time_rate"] = round(100 * clean.on_time.mean(), 2)
OUT["avg_delay"] = round(clean.delay_days.mean(), 2)
OUT["avg_delay_late_only"] = round(clean.loc[clean.delay_days > 0, "delay_days"].mean(), 2)
OUT["total_cost"] = float(clean.transport_cost.sum())
OUT["cost_per_kg_overall"] = round(clean.transport_cost.sum() / clean.weight_kg.sum(), 2)

# Group stats
def grp(col):
    g = clean.groupby(col).agg(
        shipments=("shipment_id", "count"),
        otd=("on_time", "mean"),
        avg_delay=("delay_days", "mean"),
        avg_actual=("actual_days", "mean"),
        avg_cost=("transport_cost", "mean"),
        cost_kg=("cost_per_kg", "median"),
        cost_tkm=("cost_per_tkm", "median"),
        total_cost=("transport_cost", "sum"),
    )
    g["otd"] = (g.otd * 100).round(1)
    return g.round(2)

for c in ["carrier", "mode", "origin_hub", "dest_region", "quarter", "priority"]:
    OUT["grp_" + c] = grp(c).to_dict("index")
OUT["grp_month"] = grp("month").to_dict("index")

# Correlation
corr_cols = ["distance_km", "weight_kg", "volume_m3", "units",
             "transport_cost", "planned_days", "actual_days", "delay_days"]
corr = clean[corr_cols].corr(method="pearson")
OUT["corr_cost"] = corr["transport_cost"].round(3).to_dict()
OUT["corr_delay"] = corr["delay_days"].round(3).to_dict()
spear = clean[corr_cols].corr(method="spearman")
OUT["spearman_cost"] = spear["transport_cost"].round(3).to_dict()

# Statistical tests
groups = [g.delay_days.values for _, g in clean.groupby("carrier")]
kw = stats.kruskal(*groups)
OUT["kruskal_carrier_delay"] = {"H": round(float(kw.statistic), 2), "p": float(kw.pvalue)}
peak = clean[clean.month.isin([10, 11, 12])].delay_days
offp = clean[~clean.month.isin([10, 11, 12])].delay_days
tt = stats.ttest_ind(peak, offp, equal_var=False)
OUT["ttest_peak"] = {"peak_mean": round(float(peak.mean()), 3), "off_mean": round(float(offp.mean()), 3),
                     "t": round(float(tt.statistic), 2), "p": float(tt.pvalue),
                     "n_peak": int(len(peak)), "n_off": int(len(offp))}
# normality of delivery time
sh = stats.shapiro(clean.actual_days.sample(5000, random_state=1))
OUT["shapiro_actual_days"] = {"W": round(float(sh.statistic), 4), "p": float(sh.pvalue)}

# Regression cost ~ tonne-km  (Road only, for worked example)
road = clean[clean["mode"] == "Road"].copy()
road["tkm"] = np.maximum(road.weight_kg, 250) / 1000 * road.distance_km
sl = stats.linregress(road.tkm, road.transport_cost)
OUT["reg_road"] = {"slope": round(sl.slope, 3), "intercept": round(sl.intercept, 1),
                   "r2": round(sl.rvalue ** 2, 4), "n": int(len(road))}

# Lane analysis
lane = clean.groupby("lane").agg(n=("shipment_id", "count"), otd=("on_time", "mean"),
                                  delay=("delay_days", "mean")).reset_index()
lane = lane[lane.n >= 30].sort_values("delay", ascending=False)
OUT["worst_lanes"] = lane.head(5).round(3).to_dict("records")
OUT["best_lanes"] = lane.tail(3).round(3).to_dict("records")

# ----------------------------------------------------------------------
# 4. VISUALISATIONS
# ----------------------------------------------------------------------
# Fig 1 - distribution of actual delivery time
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(clean.actual_days, bins=range(1, int(clean.actual_days.max()) + 2), ax=ax[0], color="#2a6f97")
ax[0].axvline(clean.actual_days.mean(), color="red", ls="--", label=f"Mean {clean.actual_days.mean():.2f}")
ax[0].axvline(clean.actual_days.median(), color="green", ls=":", label=f"Median {clean.actual_days.median():.0f}")
ax[0].set(title="Distribution of Actual Delivery Time", xlabel="Days", ylabel="Shipments")
ax[0].legend()
sns.histplot(clean.delay_days, bins=range(int(clean.delay_days.min()), int(clean.delay_days.max()) + 2),
             ax=ax[1], color="#e76f51")
ax[1].axvline(0, color="black", lw=1)
ax[1].set(title="Delay vs Plan (days; >0 = late)", xlabel="Delay (days)", ylabel="Shipments")
plt.tight_layout(); plt.savefig(FIG + "fig1_dist.png", dpi=160); plt.close()

# Fig 2 - boxplot delay by carrier + OTD annotation
fig, ax = plt.subplots(figsize=(8, 4.5))
order = clean.groupby("carrier").delay_days.mean().sort_values().index
sns.boxplot(data=clean, x="carrier", y="delay_days", order=order, ax=ax, showfliers=False)
for i, c in enumerate(order):
    ax.text(i, clean.delay_days.quantile(0.985) + 0.2,
            f"OTD {clean[clean.carrier==c].on_time.mean()*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set(title="Delay Days by Carrier", xlabel="", ylabel="Delay (days)")
plt.tight_layout(); plt.savefig(FIG + "fig2_carrier_box.png", dpi=160); plt.close()

# Fig 3 - monthly volume vs OTD
m = clean.groupby("month").agg(n=("shipment_id", "count"), otd=("on_time", "mean")).reset_index()
fig, ax1 = plt.subplots(figsize=(10, 4.5))
ax1.bar(m.month, m.n, color="#8ecae6", label="Shipments")
ax1.set_xlabel("Month (2025)"); ax1.set_ylabel("Shipments"); ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(m.month, m.otd * 100, color="#d00000", marker="o", lw=2, label="On-time %")
ax2.set_ylabel("On-time delivery %"); ax2.grid(False)
ax2.set_ylim(40, 80)
for x, y in zip(m.month, m.otd * 100):
    ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#d00000")
ax1.set_title("Monthly Shipment Volume vs On-Time Delivery Rate")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
plt.tight_layout(); plt.savefig(FIG + "fig3_monthly.png", dpi=160); plt.close()

# Fig 4 - correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6.2))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, mask=mask, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Matrix")
plt.tight_layout(); plt.savefig(FIG + "fig4_corr.png", dpi=160); plt.close()

# Fig 5 - cost vs distance scatter by mode (log-y)
fig, ax = plt.subplots(figsize=(8.5, 5))
samp = clean.sample(2000, random_state=3)
sns.scatterplot(data=samp, x="distance_km", y="transport_cost", hue="mode", size="weight_kg",
                sizes=(8, 140), alpha=0.5, ax=ax)
ax.set_yscale("log")
ax.set(title="Transport Cost vs Distance (bubble size = weight)", xlabel="Distance (km)", ylabel="Cost (INR, log scale)")
plt.tight_layout(); plt.savefig(FIG + "fig5_cost_scatter.png", dpi=160); plt.close()

# Fig 6 - cost per kg by mode and weight band
clean["weight_band"] = pd.cut(clean.weight_kg, [0, 150, 400, 1000, 3000, 10000],
                              labels=["<150", "150-400", "400-1000", "1000-3000", ">3000"])
pv = clean.pivot_table(index="weight_band", columns="mode", values="cost_per_kg", aggfunc="median", observed=True)
OUT["cost_per_kg_band_mode"] = pv.round(2).to_dict()
fig, ax = plt.subplots(figsize=(8.5, 4.5))
pv.plot(kind="bar", ax=ax, rot=0)
ax.set(title="Median Cost per kg by Weight Band and Mode", xlabel="Weight band (kg)", ylabel="INR per kg")
plt.tight_layout(); plt.savefig(FIG + "fig6_costkg.png", dpi=160); plt.close()

# Fig 7 - hub x region delay heatmap (bottleneck)
hm = clean.pivot_table(index="origin_hub", columns="dest_region", values="delay_days", aggfunc="mean")
fig, ax = plt.subplots(figsize=(7.5, 4.8))
sns.heatmap(hm, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean delay (days)"})
ax.set_title("Mean Delay: Origin Hub x Destination Region"); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG + "fig7_heatmap.png", dpi=160); plt.close()

# Fig 8 - cost share vs OTD by carrier (bar + table style) -> avg cost per kg by carrier & mode (mix check)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
cm = clean.groupby(["carrier", "mode"]).cost_per_tkm.median().unstack()
cm.plot(kind="bar", ax=ax, rot=0)
ax.set(title="Median Cost per Tonne-km by Carrier and Mode", xlabel="", ylabel="INR / tonne-km")
plt.tight_layout(); plt.savefig(FIG + "fig8_carrier_cost.png", dpi=160); plt.close()
OUT["carrier_mode_tkm"] = cm.round(2).to_dict()

# ----------------------------------------------------------------------
# 5. RECOMMENDATION IMPACT (worked numbers)
# ----------------------------------------------------------------------
cp = clean[clean.carrier == "CargoPlus"]
ss = clean[clean.carrier == "SwiftLine"]
# scenario A: shift half of CargoPlus Road/Rail volume to the best-performing carriers
shift_n = int(len(cp) * 0.5)
otd_cp, otd_rest = cp.on_time.mean(), clean[clean.carrier != "CargoPlus"].on_time.mean()
new_otd = (clean.on_time.sum() + shift_n * (otd_rest - otd_cp)) / len(clean)
OUT["scenario_shift"] = {"cp_n": int(len(cp)), "shift_n": shift_n,
                         "otd_cp": round(otd_cp * 100, 1), "otd_rest": round(otd_rest * 100, 1),
                         "new_overall_otd": round(new_otd * 100, 2)}
# scenario B: Air -> Express-road/rail for Standard priority, distance<=1500 km
air_std = clean[(clean["mode"] == "Air") & (clean.priority == "Standard")]
air_short = air_std[air_std.distance_km <= 1500]
road_med_tkm = clean[(clean["mode"] == "Road")].cost_per_tkm.median()
air_med_tkm = clean[(clean["mode"] == "Air")].cost_per_tkm.median()
est_save = float(air_short.transport_cost.sum() * (1 - road_med_tkm / air_med_tkm))
OUT["scenario_air"] = {"n": int(len(air_short)), "air_cost": float(air_short.transport_cost.sum()),
                       "road_over_air_ratio": round(road_med_tkm / air_med_tkm, 3),
                       "est_saving": round(est_save, 0),
                       "pct_of_total": round(100 * est_save / OUT["total_cost"], 2),
                       "extra_days_avg": round(float((np.ceil(air_short.distance_km / 520 + 1) -
                                                      np.ceil(air_short.distance_km / 1800 + 1)).mean()), 2)}
# scenario C: peak-season uplift -> reduce Q4 delay by pre-booking: 30% reduction of Q4 delay-days
q4 = clean[clean.month.isin([10, 11, 12])]
OUT["scenario_peak"] = {"q4_n": int(len(q4)), "q4_otd": round(q4.on_time.mean() * 100, 1),
                        "other_otd": round(clean[~clean.month.isin([10, 11, 12])].on_time.mean() * 100, 1),
                        "q4_share_pct": round(100 * len(q4) / len(clean), 1)}
# Kolkata -> North
kn = clean[(clean.origin_hub == "Kolkata") & (clean.dest_region == "North")]
OUT["kolkata_north"] = {"n": int(len(kn)), "otd": round(kn.on_time.mean() * 100, 1),
                        "avg_delay": round(kn.delay_days.mean(), 2),
                        "other_otd": round(clean.drop(kn.index).on_time.mean() * 100, 1)}

# Worked example rows
road_ex = clean[(clean["mode"] == "Road") & (clean.carrier == "SwiftLine") & (clean.priority == "Standard")]
ex = road_ex.iloc[(road_ex.distance_km - 1400).abs().argsort()[:1]].iloc[0]
OUT["example_row"] = {k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else str(v))
                      for k, v in ex[["shipment_id", "origin_hub", "dest_region", "carrier", "mode", "distance_km",
                                      "weight_kg", "transport_cost", "planned_days", "actual_days",
                                      "delay_days", "cost_per_tkm", "cost_per_kg"]].items()}

with open("results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print(json.dumps(OUT, indent=1, default=str)[:12000])

# ----------------------------------------------------------------------
# 6. EXTRA VALIDATION + SUPPORTING NUMBERS FOR THE REPORT
# ----------------------------------------------------------------------
R2 = {}
n = len(clean)
# cross-field consistency checks on the cleaned data
R2["consistency"] = {
    "actual_eq_planned_plus_delay": int((clean.actual_days == clean.planned_days + clean.delay_days).sum()),
    "on_time_flag_consistent": int((clean.on_time == (clean.delay_days <= 0).astype(int)).sum()),
    "density_kg_m3_min": round(float((clean.weight_kg / clean.volume_m3).min()), 1),
    "density_kg_m3_max": round(float((clean.weight_kg / clean.volume_m3).max()), 1),
    "date_min": str(clean.ship_date.min().date()), "date_max": str(clean.ship_date.max().date()),
    "mode_values": sorted(clean["mode"].unique().tolist()),
    "carrier_values": sorted(clean.carrier.unique().tolist()),
    "min_distance": float(clean.distance_km.min()), "max_distance": float(clean.distance_km.max()),
    "min_cost": float(clean.transport_cost.min()), "max_cost": float(clean.transport_cost.max()),
    "n": n,
}
# Wilson 95% CI for OTD
def wilson(k, n_, z=1.96):
    p = k / n_
    d = 1 + z**2 / n_
    c = (p + z**2 / (2 * n_)) / d
    h = z * np.sqrt(p * (1 - p) / n_ + z**2 / (4 * n_**2)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)
R2["otd_ci_overall"] = wilson(int(clean.on_time.sum()), n)
R2["otd_on_time_count"] = int(clean.on_time.sum())
for c in clean.carrier.unique():
    s_ = clean[clean.carrier == c]
    R2.setdefault("otd_ci_carrier", {})[c] = {"on_time": int(s_.on_time.sum()), "n": int(len(s_)),
                                              "ci": wilson(int(s_.on_time.sum()), len(s_))}
ct = pd.crosstab(clean.carrier, clean.on_time)
chi = stats.chi2_contingency(ct)
R2["chi2_carrier_otd"] = {"chi2": round(float(chi[0]), 1), "p": float(chi[1]), "dof": int(chi[2])}
# spend concentration
mode_spend = clean.groupby("mode").transport_cost.sum()
R2["air_spend_share_pct"] = round(100 * mode_spend["Air"] / mode_spend.sum(), 1)
R2["air_shipment_share_pct"] = round(100 * (clean["mode"] == "Air").mean(), 1)
# roads regression worked prediction
slope, icpt = OUT["reg_road"]["slope"], OUT["reg_road"]["intercept"]
R2["pred_1000kg_1500km_road"] = round(icpt + slope * (1.0 * 1500), 0)
# Kolkata scenario
kol = clean[clean.origin_hub == "Kolkata"]
R2["kolkata"] = {"n": int(len(kol)), "otd": round(100 * kol.on_time.mean(), 1),
                 "gain_if_network_avg": round(float(len(kol) * (clean.on_time.mean() - kol.on_time.mean())), 0)}
R2["kolkata"]["pts"] = round(100 * R2["kolkata"]["gain_if_network_avg"] / n, 2)
# peak scenario: recover half of the Q4 gap
gap = (clean[~clean.month.isin([10, 11, 12])].on_time.mean() - q4.on_time.mean())
R2["peak"] = {"gap_pts": round(100 * gap, 1), "gain_shipments": round(float(len(q4) * gap / 2), 0),
              "gain_pts": round(100 * len(q4) * gap / 2 / n, 2)}
# carrier shift gain
cp_gain = shift_n * (otd_rest - otd_cp)
R2["shift"] = {"gain_shipments": round(float(cp_gain), 0), "gain_pts": round(100 * cp_gain / n, 2),
               "cp_on_time": int(cp.on_time.sum())}
# light-shipment consolidation
light = clean[(clean.weight_kg < 150) & (clean["mode"] != "Air")]
med_light = light.cost_per_kg.median()
mid = clean[(clean.weight_kg >= 150) & (clean.weight_kg < 400) & (clean["mode"] != "Air")].cost_per_kg.median()
sav = light.weight_kg.sum() * (med_light - mid) * 0.5
R2["light"] = {"n": int(len(light)), "share_pct": round(100 * len(light) / n, 1),
               "spend": float(light.transport_cost.sum()),
               "median_cost_kg": round(float(med_light), 2), "mid_cost_kg": round(float(mid), 2),
               "total_kg": round(float(light.weight_kg.sum()), 0),
               "saving_50pct": round(float(sav), 0), "saving_pct_total": round(100 * sav / OUT["total_cost"], 2)}
# sensitivity for air->road scenario
air_cost = OUT["scenario_air"]["air_cost"]
R2["air_sens"] = {r: round(air_cost * (1 - r), 0) for r in (0.257, 0.40, 0.50)}
R2["air_sens_pct"] = {r: round(100 * air_cost * (1 - r) / OUT["total_cost"], 2) for r in (0.257, 0.40, 0.50)}
# sample rows for the report table
smp = clean[["shipment_id", "origin_hub", "dest_region", "carrier", "mode", "distance_km", "weight_kg",
             "transport_cost", "planned_days", "actual_days", "delay_days"]].head(8)
R2["sample"] = smp.to_dict("records")
R2["worked_ex_tkm"] = round(max(OUT["example_row"]["weight_kg"], 250) / 1000 * OUT["example_row"]["distance_km"], 2)
with open("results2.json", "w") as f:
    json.dump(R2, f, indent=2, default=str)
print(json.dumps(R2, indent=1, default=str))
