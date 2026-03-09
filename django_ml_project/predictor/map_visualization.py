import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def _compute_centroids(geojson: Dict[str, Any]) -> pd.DataFrame:
    """Compute simple centroids (mean lat/lon) for each district polygon."""
    centroids: List[Dict[str, Any]] = []

    for feature in geojson["features"]:
        name = feature["properties"]["NAME_2"].strip()
        feature["id"] = name
        coords = feature["geometry"]["coordinates"]

        all_lons: List[float] = []
        all_lats: List[float] = []

        def extract_coords(items):
            for item in items:
                if isinstance(item[0], (int, float, float)):
                    all_lons.append(item[0])
                    all_lats.append(item[1])
                else:
                    extract_coords(item)

        extract_coords(coords)

        if all_lons and all_lats:
            centroids.append(
                {
                    "district": name,
                    "lat": float(np.mean(all_lats)),
                    "lon": float(np.mean(all_lons)),
                }
            )

    return pd.DataFrame(centroids)


def create_rwanda_district_map(df: pd.DataFrame) -> str:
    """
    Create a Rwanda map showing vehicle client distribution by district with proper
    boundaries using GeoJSON and Plotly Mapbox choropleth, with static labels.
    """
    # Count clients per district
    district_counts = df["district"].value_counts().reset_index()
    district_counts.columns = ["district", "client_count"]
    district_counts["district"] = district_counts["district"].str.strip()

    # Load GeoJSON file (stored under dummy-data)
    geojson_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dummy-data",
        "rwanda_districts.geojson",
    )
    with open(geojson_path, "r", encoding="utf-8") as f:
        rwanda_geojson = json.load(f)

    # Compute centroids for labels
    centroid_df = _compute_centroids(rwanda_geojson)

    # Merge counts with centroids
    label_df = pd.merge(centroid_df, district_counts, on="district", how="left")
    label_df["client_count"] = label_df["client_count"].fillna(0).astype(int)
    label_df["text"] = label_df["district"] + " " + label_df["client_count"].astype(
        str
    )

    # Choropleth map
    fig = px.choropleth_mapbox(
        district_counts,
        geojson=rwanda_geojson,
        locations="district",
        color="client_count",
        # Pink / magenta-ish continuous scale
        color_continuous_scale="RdPu",
        mapbox_style="carto-positron",
        center={"lat": -1.94, "lon": 30.06},
        zoom=7.8,
        opacity=0.6,
        labels={"client_count": "Total Clients"},
    )

    # Text labels
    fig.add_trace(
        go.Scattermapbox(
            lat=label_df["lat"],
            lon=label_df["lon"],
            mode="text",
            text=label_df["text"],
            textfont={"size": 10, "color": "black"},
            hoverinfo="none",
            showlegend=False,
        )
    )

    fig.update_traces(
        marker_line_width=1,
        marker_line_color="#7a0177",  # dark pink border
        selector=dict(type="choroplethmapbox"),
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_mapboxes(center={"lat": -1.94, "lon": 30.06}, zoom=7.8)

    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def get_district_summary_table(df: pd.DataFrame) -> str:
    """Summary table of clients by district (and province when available)."""
    if "province" in df.columns:
        group_cols = ["province", "district"]
        summary = (
            df.groupby(group_cols)
            .agg(
                {
                    "client_name": "count",
                    "estimated_income": "mean",
                    "selling_price": "mean",
                }
            )
            .reset_index()
        )
        summary.columns = [
            "Province",
            "District",
            "Number of Clients",
            "Avg Income",
            "Avg Price",
        ]
        sort_cols = ["Province", "District"]
    else:
        group_cols = ["district"]
        summary = (
            df.groupby(group_cols)
            .agg(
                {
                    "client_name": "count",
                    "estimated_income": "mean",
                    "selling_price": "mean",
                }
            )
            .reset_index()
        )
        summary.columns = [
            "District",
            "Number of Clients",
            "Avg Income",
            "Avg Price",
        ]
        sort_cols = ["District"]
    summary["Avg Income"] = summary["Avg Income"].round(2)
    summary["Avg Price"] = summary["Avg Price"].round(2)
    summary = summary.sort_values(sort_cols)

    return summary.to_html(
        classes="table table-bordered table-striped table-sm",
        index=False,
        justify="center",
    )

