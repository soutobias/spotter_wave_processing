import xarray as xr

files = [
    "2021_10.grib",
    "2021_11.grib",
    "2021_12.grib",
    "2022_01.grib",
    "2022_02.grib",
    "2022_03.grib",
]

lat_min, lat_max = -67.5, -50.5
lon_min, lon_max = -75.5, -30.5

for file in files:
    ds = xr.open_dataset(
        file,
        engine="cfgrib",
        chunks={
            "time": 1,
            "directionNumber": 24,
            "frequencyNumber": 30,
        },
    )

    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180) % 360) - 180)
        )

    mask = (
        (ds.latitude >= lat_min) &
        (ds.latitude <= lat_max) &
        (ds.longitude >= lon_min) &
        (ds.longitude <= lon_max)
    ).compute()
    ds_box = ds.isel(values=mask)

    print(ds_box.sizes)

    ds_box = ds_box.chunk({
        "time": 1,
        "directionNumber": 24,
        "frequencyNumber": 30,
        "values": -1,
    })
