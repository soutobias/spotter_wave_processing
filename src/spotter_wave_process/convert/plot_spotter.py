import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import folium
import branca.colormap as cm
import pandas as pd

def plot_all_spotter_points_by_time(
    ds,
    figsize=(12, 8),
    point_size=2,
    alpha=0.6,
    zoom=True,
    margin=2,
):
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    time = ds["time"].values

    valid = (
        ~np.isnan(lat)
        & ~np.isnan(lon)
        & (lat >= -90)
        & (lat <= 90)
        & (lon >= -180)
        & (lon <= 180)
        & ~np.isnat(time)
    )

    lat = lat[valid]
    lon = lon[valid]
    time = time[valid]

    # Convert datetime64 to matplotlib numeric dates
    time_numeric = mdates.date2num(time.astype("datetime64[ms]").astype(object))

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    if zoom:
        ax.set_extent(
            [
                lon.min() - margin,
                lon.max() + margin,
                lat.min() - margin,
                lat.max() + margin,
            ],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_global()

    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    sc = ax.scatter(
        lon,
        lat,
        c=time_numeric,
        s=point_size,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
    cbar.set_label("Time")

    # Format colorbar ticks as dates
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax.set_title("Spotter Buoy Positions Colored by Time")

    return fig, ax


def plot_all_spotter_points_by_time_folium(
    ds,
    step=5,
    point_radius=2,
    fill_opacity=0.7,
    tiles="CartoDB positron",
    output_html=None,
):
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    time = ds["time"].values

    valid = (
        ~np.isnan(lat)
        & ~np.isnan(lon)
        & (lat >= -90)
        & (lat <= 90)
        & (lon >= -180)
        & (lon <= 180)
        & ~np.isnat(time)
    )

    lat = lat[valid][::step]
    lon = lon[valid][::step]
    time = time[valid][::step]

    time_pd = pd.to_datetime(time)

    time_numeric = time_pd.astype("int64") / 1e9

    m = folium.Map(
        location=[float(np.nanmean(lat)), float(np.nanmean(lon))],
        zoom_start=4,
        tiles=tiles,
    )

    colormap = cm.LinearColormap(
        colors=["blue", "cyan", "yellow", "orange", "red"],
        vmin=float(time_numeric.min()),
        vmax=float(time_numeric.max()),
        caption="Time",
    )

    for la, lo, t, tn in zip(lat, lon, time_pd, time_numeric):
        folium.CircleMarker(
            location=[float(la), float(lo)],
            radius=point_radius,
            color=colormap(float(tn)),
            fill=True,
            fill_color=colormap(float(tn)),
            fill_opacity=fill_opacity,
            weight=0,
            popup=str(t),
        ).add_to(m)

    colormap.add_to(m)

    if output_html is not None:
        m.save(output_html)

    return m
