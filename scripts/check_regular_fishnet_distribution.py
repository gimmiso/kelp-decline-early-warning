from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

geojson_path = Path("kelpwatch_regular_10km_fishnet_preview.geojson")
gdf = gpd.read_file(geojson_path)
gdf_proj = gdf.to_crs("EPSG:3310")

overlaps = []
touch_pairs = 0

for i in range(len(gdf_proj)):
    for j in range(i + 1, len(gdf_proj)):
        area = gdf_proj.geometry.iloc[i].intersection(gdf_proj.geometry.iloc[j]).area
        if area > 1e-6:
            overlaps.append((gdf_proj.cell_id.iloc[i], gdf_proj.cell_id.iloc[j], area))
        if gdf_proj.geometry.iloc[i].touches(gdf_proj.geometry.iloc[j]):
            touch_pairs += 1

print(f"Total cells: {len(gdf)}")
print(f"Positive-area overlaps: {len(overlaps)}")
print(f"Boundary-touching pairs: {touch_pairs}")

if overlaps:
    for a, b, area in overlaps:
        print(f"{a} overlaps {b}: {area:.2f} m²")
else:
    print("No positive-area overlaps detected.")

ax = gdf.plot(
    column="region_group",
    alpha=0.25,
    edgecolor="black",
    linewidth=0.5,
    figsize=(8, 12),
    legend=True
)

for _, row in gdf.iterrows():
    c = row.geometry.centroid
    ax.annotate(row["cell_id"].replace("cell_", ""), (c.x, c.y), fontsize=5, ha="center", va="center")

plt.title("Regular 10 km Fishnet AOIs for Kelpwatch")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("regular_10km_fishnet_static_preview.png", dpi=300)
plt.show()