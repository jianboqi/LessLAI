import numpy as np
from osgeo import gdal
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ----------------------------- Input parameters -----------------------------
input_tif  = r'Landsat_0618(4bands).tif'   # 3 bands
output_tif = r'lai_0618_stand.tif'

ebf_lut_file = r'luts/EBF_lut_l8.npy'
dbf_lut_file = r'luts/DBF_lut_l8.npy'
enf_lut_file = r'luts/ENF_lut_l8.npy'
dnf_lut_file = r'luts/DNF_lut_l8.npy'
ndvi_lai_lut_file = r'luts/backup.npy'

sza = 25

# ----------Constants ----------
LAI_COL, SZA_COL, RED_COL, NIR_COL = 0, 1, 2, 3

# ----------------------------- Load LUTs -----------------------------
def load_and_filter(npy_path, sza) -> np.ndarray:
    """
    Find the angle in the LUT that is closest to the input sza,
    and return all samples for that angle.
    """
    data = np.load(npy_path, allow_pickle=True)      # shape (N, >=3)
    # Column 2 holds the angles; extract unique values and sort
    uniq_ang = np.unique(data[:, SZA_COL])
    # Locate the nearest angle
    nearest_ang = uniq_ang[np.argmin(np.abs(uniq_ang - sza))]
    # Extract all samples for that angle
    return data[data[:, SZA_COL] == nearest_ang]

ebf_lut = load_and_filter(ebf_lut_file, sza)
dbf_lut = load_and_filter(dbf_lut_file, sza)
enf_lut = load_and_filter(enf_lut_file, sza)
dnf_lut = load_and_filter(dnf_lut_file, sza)
ndvi_lai_lut = np.load(ndvi_lai_lut_file)

# ----------------------------- Row-wise processor -----------------------------
def process_row(i, red_band, nir_band, landuse_band, target_codes,
                ebf_lut, dbf_lut, enf_lut, dnf_lut, ndvi_lai_lut):
    ncols = red_band.shape[1]
    out_row = np.full(ncols, -9999.0, dtype=np.float32)
    for j in range(ncols):
        code = landuse_band[i, j]
        if code not in target_codes:
            continue

        # Select corresponding LUT
        if code in [51, 52]:
            ref_data, biome_code = ebf_lut, 5
        elif code in [61, 62]:
            ref_data, biome_code = dbf_lut, 6
        elif code in [71, 72]:
            ref_data, biome_code = enf_lut, 7
        elif code in [81, 82]:
            ref_data, biome_code = dnf_lut, 8
        elif code in [91, 92]:
            ref_data, biome_code = np.vstack([ebf_lut, dbf_lut, enf_lut, dnf_lut]), 9
        else:
            continue

        pixel_red = red_band[i, j]
        pixel_nir = nir_band[i, j]
        if pixel_red == 0 or pixel_nir == 0:
            continue

        # Compute NDVI on-the-fly
        ndvi_val = (pixel_nir - pixel_red) / (pixel_nir + pixel_red + 1e-6)

        # LUT matching
        solutions = []
        for row_data in ref_data:
            lai_val   = row_data[LAI_COL]
            ref_red   = row_data[RED_COL]
            ref_nir   = row_data[NIR_COL]
            denom_nir = (0.15 * pixel_nir) ** 2
            denom_red = (0.30 * pixel_red) ** 2
            delta_square = ((pixel_nir - ref_nir) ** 2) / denom_nir + \
                           ((pixel_red - ref_red) ** 2) / denom_red
            if delta_square <= 2:
                solutions.append(lai_val)

        if solutions:
            out_row[j] = np.mean(solutions)
        else:
            # NDVI fallback
            lut_subset = ndvi_lai_lut[ndvi_lai_lut[:, 0] == biome_code]
            for k in range(len(lut_subset) - 1):
                ndvi_lower = lut_subset[k, 1]
                ndvi_upper = lut_subset[k + 1, 1]
                lai_val    = lut_subset[k + 1, 2]
                if ndvi_lower <= ndvi_val < ndvi_upper:
                    out_row[j] = lai_val
                    break
    return i, out_row

# ----------------------------- Main -----------------------------
def main():
    ds = gdal.Open(input_tif)
    if ds is None:
        sys.exit(f"Cannot open input TIF: {input_tif}")

    cols = ds.RasterXSize
    rows = ds.RasterYSize
    red_band  = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nir_band  = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    landuse_band = ds.GetRasterBand(3).ReadAsArray()

    output_arr = np.full((rows, cols), -9999.0, dtype=np.float32)
    target_codes = [51, 52, 61, 62, 71, 72, 81, 82, 91, 92]

    with tqdm_joblib(tqdm(desc="Processing rows", total=rows)):
        results = Parallel(n_jobs=10)(
            delayed(process_row)(i, red_band, nir_band, landuse_band, target_codes,
                                 ebf_lut, dbf_lut, enf_lut, dnf_lut, ndvi_lai_lut)
            for i in range(rows)
        )
    for i, out_row in results:
        output_arr[i, :] = out_row

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(output_arr)
    band.SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    del out_ds
    print("Processing complete ->", output_tif)

if __name__ == '__main__':
    main()