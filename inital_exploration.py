# grocery_graphs.py
# Create 3 visuals from grocerydb.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 1) Load data
csv_path = Path("project1/grocerydb.csv")   # change if your path differs
df = pd.read_csv(csv_path)
df = df.rename(columns={c: c.strip() for c in df.columns})  # trim spaces

# Helper: find the "price per calorie" column even if named slightly differently
ppc_candidates = [c for c in df.columns if c.lower().replace(" ", "") in {
    "pricepercal", "price_percal", "pricepercalorie"
}]
price_percal = ppc_candidates[0] if ppc_candidates else "price percal"

# -------------------------------
# PLOT 1: Value by Category (boxplot of price per calorie for top categories)
# -------------------------------
top_cats = df["category"].value_counts().head(10).index.tolist()
sub1 = df[df["category"].isin(top_cats) & df[price_percal].notna()].copy()

# order categories by median price_percal (best value on the left)
cat_order = (sub1.groupby("category")[price_percal]
             .median()
             .sort_values(ascending=True)
             .index
             .tolist())

plt.figure(figsize=(10, 6))
data = [sub1.loc[sub1["category"] == cat, price_percal].values for cat in cat_order]
plt.boxplot(data, labels=cat_order, showfliers=False)
plt.xticks(rotation=30, ha='right')
plt.ylabel("Price per Calorie")
plt.title("Value by Category: Distribution of Price per Calorie (Top 10 Categories)")
out1 = Path("project1/images/plot_value_by_category_boxplot.png")
plt.tight_layout()
plt.savefig(out1)

# -------------------------------
# PLOT 2: Processing vs Macros (grouped bar chart)
# -------------------------------
macros = ["Protein", "Total Fat", "Carbohydrate", "Sugars, total"]
avail_macros = [m for m in macros if m in df.columns]
sub2 = df[df["FPro_class"].notna()].copy()
macro_means = (sub2
               .groupby("FPro_class")[avail_macros]
               .mean()
               .reindex(sorted(sub2["FPro_class"].dropna().unique()))
               )

x = np.arange(len(avail_macros))
width = 0.18 if len(macro_means.index) > 0 else 0.2

plt.figure(figsize=(10, 6))
for i, cls in enumerate(macro_means.index):
    plt.bar(x + i*width, macro_means.loc[cls].values,
            width=width, label=f"FPro Class {int(cls)}")
plt.xticks(x + width*(len(macro_means.index)-1)/2, avail_macros)
plt.ylabel("Average per 100g (normalized units)")
plt.title("Processing vs Nutrition: Average Macros by FPro Class")
plt.legend()
out2 = Path("project1/images/plot_processing_vs_macros_groupedbar.png")
plt.tight_layout()
plt.savefig(out2)

# -------------------------------
# PLOT 3: Size vs Price (hexbin density)
# -------------------------------
sub3 = df[["package_weight", "price"]].dropna()
plt.figure(figsize=(9, 7))
hb = plt.hexbin(sub3["package_weight"], sub3["price"], gridsize=40, mincnt=5)
plt.xlabel("Package Weight")
plt.ylabel("Price")
plt.title("Size vs Price: Density of Package Weight vs Price")
cb = plt.colorbar(hb)
cb.set_label("Count")
out3 = Path("project1/images/plot_size_vs_price_hexbin.png")
plt.tight_layout()
plt.savefig(out3)

print("Saved plots:")
print(out1)
print(out2)
print(out3)


# sugar_protein_bubble.py
# Zoomed log–log bubble scatter: Sugar vs Protein, colored by category, bubble size = price



# Keep top categories so the plot stays readable
top_categories = df["category"].value_counts().head(8).index
plot_df = df[df["category"].isin(top_categories)].copy()

# Require sugar, protein, and price to be present and positive for log-log + bubble size
plot_df = plot_df[
    plot_df["Sugars, total"].notna() &
    plot_df["Protein"].notna() &
    plot_df["price"].notna()
].copy()

# Avoid zeros on log scale; drop rows with nonpositive values
plot_df = plot_df[(plot_df["Sugars, total"] > 0) & (plot_df["Protein"] > 0)]

# --- Bubble size scaling (gentle) ---
def scale_sizes(prices, min_size=20, max_size=200, low_q=5, high_q=95):
    """Map prices to marker sizes with quantile clipping for stability."""
    p = prices.to_numpy(dtype=float)
    lo = np.nanpercentile(p, low_q)
    hi = np.nanpercentile(p, high_q)
    p = np.clip(p, lo, hi)
    if hi == lo:  # edge case: all same
        return np.full_like(p, (min_size + max_size) / 2.0)
    norm = (p - lo) / (hi - lo)
    return min_size + norm * (max_size - min_size)

plot_df["bubble_size"] = scale_sizes(plot_df["price"], min_size=20, max_size=200)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 7))

# Color cycle from matplotlib defaults
cats = list(top_categories)
colors = plt.cm.tab10.colors if len(cats) <= 10 else plt.cm.tab20.colors

for i, cat in enumerate(cats):
    sub = plot_df[plot_df["category"] == cat]
    ax.scatter(
        sub["Sugars, total"], sub["Protein"],
        #s=sub["bubble_size"],
        label=cat,
        alpha=0.7
        # no explicit colors per tool guidance; matplotlib uses cycle
    )

# Log scales + zoomed view (0.1 to 100)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1, 120)
ax.set_ylim(0.1, 120)

ax.set_xlabel("Sugar (log scale)")
ax.set_ylabel("Protein (log scale)")
ax.set_title("Sugar vs Protein (Log Scale)")

# Legend outside the plot area
ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

plt.tight_layout()

# Save figure
out_path = Path("project1/images/plot_sugar_vs_protein_bubble_log_zoom_matplotlib.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to: {out_path}")



# sugar_protein_violin.py
# Violin plot of sugar-to-protein ratio across top categories


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Load & prep ---
df = pd.read_csv("project1/grocerydb.csv")
df = df.rename(columns={c: c.strip() for c in df.columns})
sub = df[(df["Protein"].notna()) & (df["Protein"] > 0) & (df["Sugars, total"].notna())].copy()
sub["ratio"] = sub["Sugars, total"] / sub["Protein"]

# Top categories to reduce clutter
top_cats = sub["category"].value_counts().head(8).index
sub = sub[sub["category"].isin(top_cats)]

# Log transform (use log10; add tiny epsilon in case of tiny values)
eps = 1e-9
sub["log_ratio"] = np.log10(sub["ratio"] + eps)

# Prepare data per category
cats = list(top_cats)
# Use raw ratios, not np.log10
data = [sub.loc[sub["category"] == c, "ratio"].values for c in cats]

fig, ax = plt.subplots(figsize=(10,7))
parts = ax.violinplot(data, showmedians=True, showextrema=False)

ax.set_yscale("log")   # let matplotlib do log scaling directly
ax.set_xticks(np.arange(1, len(cats)+1))
ax.set_xticklabels(cats, rotation=30, ha="right")
ax.set_ylabel("Sugar-to-Protein Ratio (log scale)")
ax.set_title("Sugar-to-Protein Ratio by Category (log axis)")


plt.tight_layout()
plt.savefig("project1/images/violin_ratio_log.png", dpi=150)
print("Saved /project1/images/violin_ratio_log.png")

