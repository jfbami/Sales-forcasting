
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 11


DATA_DIR = "/content"
PLOT_DIR = "/content/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["date"])
stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
oil = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"), parse_dates=["date"])
holidays = pd.read_csv(
    os.path.join(DATA_DIR, "holidays_events.csv"), parse_dates=["date"]
)
transactions = pd.read_csv(
    os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["date"]
)

train = train.merge(stores, on="store_nbr", how="left")

# ─── 1. Overall Sales Trend ─────────────────────────────────────────────────
print("1. Overall sales trend...")
daily = train.groupby("date")["sales"].mean().reset_index()

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily["date"], daily["sales"], linewidth=0.6, alpha=0.7, color="steelblue")

# 30-day rolling average
daily["ma30"] = daily["sales"].rolling(30).mean()
ax.plot(daily["date"], daily["ma30"], linewidth=2, color="darkred", label="30-day MA")
ax.axvline(pd.Timestamp("2016-04-16"), color="orange", linestyle="--", label="Earthquake")
ax.set_title("Average Daily Sales Across All Stores & Families")
ax.set_xlabel("Date")
ax.set_ylabel("Avg Sales")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_overall_trend.png"), dpi=150)
plt.close()

#Day-of-Week Pattern
print("2. Day-of-week pattern...")
train["dow"] = train["date"].dt.dayofweek
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_sales = train.groupby("dow")["sales"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(dow_names, dow_sales.values, color=sns.color_palette("viridis", 7))
ax.set_title("Average Sales by Day of Week")
ax.set_ylabel("Avg Sales")
for bar, val in zip(bars, dow_sales.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{val:.0f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_day_of_week.png"), dpi=150)
plt.close()

#Monthly Seasonality 
print("3. Monthly seasonality...")
train["month"] = train["date"].dt.month
month_sales = train.groupby("month")["sales"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(1, 13), month_sales.values, color=sns.color_palette("coolwarm", 12))
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_title("Average Sales by Month")
ax.set_ylabel("Avg Sales")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_monthly_seasonality.png"), dpi=150)
plt.close()

# Year-over-Year Growth 
print("4. Year-over-year growth...")
train["year"] = train["date"].dt.year
yearly = train.groupby("year")["sales"].mean()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(yearly.index.astype(str), yearly.values,
              color=sns.color_palette("Blues_d", len(yearly)))
ax.set_title("Average Sales by Year (Growth Trend)")
ax.set_ylabel("Avg Sales")
for bar, val in zip(bars, yearly.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{val:.0f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_yearly_growth.png"), dpi=150)
plt.close()

#Top Product Families 
print("5. Product family analysis...")
family_sales = train.groupby("family")["sales"].mean().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 10))
colors = sns.color_palette("viridis", len(family_sales))
ax.barh(family_sales.index, family_sales.values, color=colors)
ax.set_title("Average Sales by Product Family")
ax.set_xlabel("Avg Sales")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_family_sales.png"), dpi=150)
plt.close()

#6. Store Type Performance ───
print("6. Store type performance...")
store_type_sales = train.groupby(["store_nbr", "type"])["sales"].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
for stype, color in zip(sorted(store_type_sales["type"].unique()),
                          sns.color_palette("Set1", 5)):
    subset = store_type_sales[store_type_sales["type"] == stype]
    ax.scatter(subset["store_nbr"], subset["sales"], label=f"Type {stype}",
               color=color, s=80, alpha=0.8, edgecolors="black", linewidth=0.5)
ax.set_title("Average Sales by Store Number (colored by Store Type)")
ax.set_xlabel("Store Number")
ax.set_ylabel("Avg Sales")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "06_store_type.png"), dpi=150)
plt.close()

# Oil Price vs Sales 
print("7. Oil price vs sales correlation...")
oil_filled = oil.set_index("date").resample("D").mean().interpolate().reset_index()
daily_oil = daily.merge(oil_filled, on="date", how="left")
daily_oil["dcoilwtico"] = daily_oil["dcoilwtico"].interpolate()

fig, ax1 = plt.subplots(figsize=(16, 5))
ax1.plot(daily_oil["date"], daily_oil["ma30"], color="steelblue",
         linewidth=1.5, label="Sales (30d MA)")
ax1.set_ylabel("Avg Sales", color="steelblue")
ax2 = ax1.twinx()
ax2.plot(daily_oil["date"], daily_oil["dcoilwtico"], color="darkorange",
         linewidth=1, alpha=0.7, label="Oil Price")
ax2.set_ylabel("Oil Price (USD)", color="darkorange")
ax1.set_title("Sales Trend vs Oil Prices (Inverse Relationship)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "07_oil_vs_sales.png"), dpi=150)
plt.close()

#8. Promotion Effect
print("8. Promotion effect...")
promo_effect = train.groupby(train["onpromotion"] > 0)["sales"].mean()

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["No Promotion", "On Promotion"], promo_effect.values,
              color=["#95a5a6", "#e74c3c"])
ax.set_title("Average Sales: Promoted vs Non-Promoted")
ax.set_ylabel("Avg Sales")
for bar, val in zip(bars, promo_effect.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "08_promotion_effect.png"), dpi=150)
plt.close()

#Earthquake Impact 
print("9. Earthquake impact analysis...")
quake = pd.Timestamp("2016-04-16")
mask = (daily["date"] >= quake - pd.Timedelta(days=30)) & \
       (daily["date"] <= quake + pd.Timedelta(days=45))
quake_df = daily[mask].copy()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(quake_df["date"], quake_df["sales"], marker="o", markersize=3,
        linewidth=1, color="steelblue")
ax.axvline(quake, color="red", linewidth=2, linestyle="--", label="Earthquake (Apr 16)")
ax.fill_betweenx([quake_df["sales"].min(), quake_df["sales"].max()],
                  quake, quake + pd.Timedelta(days=14),
                  alpha=0.15, color="red", label="Post-quake surge")
ax.set_title("Sales Around the 2016 Ecuador Earthquake")
ax.set_ylabel("Avg Daily Sales")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "09_earthquake.png"), dpi=150)
plt.close()

#Payday Effect (15th and last day of month)
print("10. Payday effect...")
train["day"] = train["date"].dt.day
train["is_month_end"] = train["date"].dt.is_month_end
train["is_15th"] = train["day"] == 15
train["is_payday"] = train["is_month_end"] | train["is_15th"]

payday_sales = train.groupby("is_payday")["sales"].mean()
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(["Normal Day", "Payday (15th/EOM)"], payday_sales.values,
              color=["#3498db", "#2ecc71"])
ax.set_title("Payday Effect on Sales")
ax.set_ylabel("Avg Sales")
for bar, val in zip(bars, payday_sales.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "10_payday_effect.png"), dpi=150)
plt.close()

#Lag Autocorrelation ─
print("11. Autocorrelation analysis...")
daily_sales = train.groupby("date")["sales"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# ACF-style manual computation
lags = range(1, 61)
autocorrs = [daily_sales.autocorr(lag=l) for l in lags]
axes[0].bar(lags, autocorrs, color="steelblue", width=0.8)
axes[0].set_title("Autocorrelation of Daily Avg Sales")
axes[0].set_xlabel("Lag (days)")
axes[0].set_ylabel("Autocorrelation")
axes[0].axhline(0, color="black", linewidth=0.5)

#Highlight key lags
for lag in [7, 14, 28]:
    axes[0].axvline(lag, color="red", alpha=0.3, linestyle="--")

#Lag scatter: today vs 7 days ago
lag7 = pd.DataFrame({"today": daily_sales.values[7:],
                      "lag7": daily_sales.values[:-7]})
axes[1].scatter(lag7["lag7"], lag7["today"], alpha=0.3, s=10, color="steelblue")
axes[1].set_title(f"Lag-7 Scatter (r={lag7['today'].corr(lag7['lag7']):.3f})")
axes[1].set_xlabel("Sales (t-7)")
axes[1].set_ylabel("Sales (t)")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "11_autocorrelation.png"), dpi=150)
plt.close()

#Transactions vs Sales 
print("12. Transactions correlation...")
daily_trans = transactions.groupby("date")["transactions"].mean().reset_index()
trans_sales = daily.merge(daily_trans, on="date", how="inner")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(trans_sales["transactions"], trans_sales["sales"],
           alpha=0.3, s=15, color="steelblue")
corr = trans_sales["transactions"].corr(trans_sales["sales"])
ax.set_title(f"Transactions vs Sales (r={corr:.3f})")
ax.set_xlabel("Avg Transactions")
ax.set_ylabel("Avg Sales")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "12_transactions_vs_sales.png"), dpi=150)
plt.close()

