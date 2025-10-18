#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LUT-based LAI retrieval (3-band input: Red/NIR/Landuse)
Usage:
    python less_lai.py -i input.tif -o lai.tif -sza 25      # GPU
    python less_lai.py -i input.tif -o lai.tif -sza 25 --cpu # CPU
    python less_lai.py -i input.tif -o lai.tif -sza 25 --cpu --sensor l8
"""
import taichi as ti
import numpy as np
import multiprocessing as mp
import argparse
import time
import sys
from osgeo import gdal

# ---------- 1. Auto backend ----------
def auto_init(force_cpu: bool = False):
    if force_cpu:
        ti.init(arch=ti.cpu, cpu_max_num_threads=mp.cpu_count())
        print("→ Forced to use CPU (LLVM) backend")
    else:
        try:
            ti.init(arch=ti.gpu)
            print("→ Using GPU backend")
        except Exception as e:
            print("GPU init failed, fall back to CPU:", e)
            ti.init(arch=ti.cpu, cpu_max_num_threads=mp.cpu_count())

# ---------- 2. Constants ----------
LAI_COL, SZA_COL, RED_COL, NIR_COL = 0, 1, 2, 3

# ---------- 3. Helper ----------
def load_and_filter(npy_path: str, sza: float) -> np.ndarray:
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

# ---------- 4. Taichi kernels (unchanged) ----------
@ti.func
def code_to_biome(code: ti.i32) -> ti.i32:
    b = 0
    if   code == 51 or code == 52: b = 5
    elif code == 61 or code == 62: b = 6
    elif code == 71 or code == 72: b = 7
    elif code == 81 or code == 82: b = 8
    elif code == 91 or code == 92: b = 9
    return b

@ti.func
def calc_ndvi(red: ti.f32, nir: ti.f32) -> ti.f32:
    return (nir - red) / (nir + red + 1e-6)

@ti.func
def add_matches(lut: ti.template(), n: ti.i32,
                pr: ti.f32, pn: ti.f32,
                denom_red: ti.f32, denom_nir: ti.f32) -> ti.types.vector(2, ti.f32):
    s, c = 0.0, 0.0
    for r in range(n):
        ref_red = lut[r, RED_COL]
        ref_nir = lut[r, NIR_COL]
        lai   = lut[r, LAI_COL]
        dsq   = ((pn - ref_nir)**2) / denom_nir + ((pr - ref_red)**2) / denom_red
        if dsq <= 2.0:
            s += lai
            c += 1.0
    return ti.Vector([s, c])

@ti.func
def ndvi_to_lai(biome: ti.i32, ndv: ti.f32, rows: ti.i32) -> ti.f32:
    res, seen, prev_ndvi = 0.0, 0, 0.0
    for r in range(rows):
        b = ti.cast(ndvi_lai_f[r, 0], ti.i32)
        if b == biome:
            cur_ndvi = ndvi_lai_f[r, 1]
            cur_lai  = ndvi_lai_f[r, 2]
            if seen == 1 and prev_ndvi <= ndv < cur_ndvi:
                res = cur_lai
                break
            prev_ndvi, seen = cur_ndvi, 1
    return res

@ti.kernel
def compute_lai(ebf_len: ti.i32, dbf_len: ti.i32,
                enf_len: ti.i32, dnf_len: ti.i32, ndvi_rows: ti.i32):
    for i, j in output_f:
        biome = code_to_biome(landuse_f[i, j])
        if biome == 0:
            output_f[i, j] = -9999.0
            continue
        pr, pn = red_f[i, j], nir_f[i, j]
        if pr == 0.0 or pn == 0.0:
            output_f[i, j] = -9999.0
            continue
        denom_nir = (0.15 * pn)**2
        denom_red = (0.30 * pr)**2
        s, c = 0.0, 0.0
        if biome == 5:
            res = add_matches(ebf_f, ebf_len, pr, pn, denom_red, denom_nir); s, c = res[0], res[1]
        elif biome == 6:
            res = add_matches(dbf_f, dbf_len, pr, pn, denom_red, denom_nir); s, c = res[0], res[1]
        elif biome == 7:
            res = add_matches(enf_f, enf_len, pr, pn, denom_red, denom_nir); s, c = res[0], res[1]
        elif biome == 8:
            res = add_matches(dnf_f, dnf_len, pr, pn, denom_red, denom_nir); s, c = res[0], res[1]
        elif biome == 9:
            res_ebf = add_matches(ebf_f, ebf_len, pr, pn, denom_red, denom_nir)
            res_dbf = add_matches(dbf_f, dbf_len, pr, pn, denom_red, denom_nir)
            res_enf = add_matches(enf_f, enf_len, pr, pn, denom_red, denom_nir)
            res_dnf = add_matches(dnf_f, dnf_len, pr, pn, denom_red, denom_nir)
            s_total = res_ebf[0] + res_dbf[0] + res_enf[0] + res_dnf[0]
            c_total = res_ebf[1] + res_dbf[1] + res_enf[1] + res_dnf[1]
            if c_total > 0.0:
                output_f[i, j] = s_total / c_total
                continue
        if c > 0.0:
            output_f[i, j] = s / c
        else:
            ndv = calc_ndvi(pr, pn)
            lai_val = ndvi_to_lai(biome, ndv, ndvi_rows)
            output_f[i, j] = lai_val if lai_val != 0.0 else -9999.0

# ---------- 5. Main workflow ----------
def run_lookup(input_tif: str, output_tif: str, sza: float, landsatx: str):
    print("Loading LUTs...")
    # Load LUTs
    lut_files = [f"luts/EBF_lut_{landsatx}.npy", f"luts/DBF_lut_{landsatx}.npy",
    f"luts/ENF_lut_{landsatx}.npy", f"luts/DNF_lut_{landsatx}.npy",
    "luts/backup.npy"]

    ebf_lut, dbf_lut, enf_lut, dnf_lut, ndvi_lai_lut = lut_files
    ebf_np = load_and_filter(ebf_lut, sza)
    dbf_np = load_and_filter(dbf_lut, sza)
    enf_np = load_and_filter(enf_lut, sza)
    dnf_np = load_and_filter(dnf_lut, sza)
    ndvi_lai_np = np.load(ndvi_lai_lut).astype(np.float32)
    ndvi_lai_np = ndvi_lai_np[np.lexsort((ndvi_lai_np[:, 1], ndvi_lai_np[:, 0]))]

    # Read 3-band image
    ds = gdal.Open(input_tif)
    if ds is None:
        sys.exit(f"Cannot open input TIF: {input_tif}")
    cols, rows = ds.RasterXSize, ds.RasterYSize
    red_np  = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nir_np  = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    land_np = ds.GetRasterBand(3).ReadAsArray().astype(np.int32)

    # Allocate fields
    global red_f, nir_f, landuse_f, output_f, ebf_f, dbf_f, enf_f, dnf_f, ndvi_lai_f
    red_f     = ti.field(ti.f32, shape=(rows, cols)); red_f.from_numpy(red_np)
    nir_f     = ti.field(ti.f32, shape=(rows, cols)); nir_f.from_numpy(nir_np)
    landuse_f = ti.field(ti.i32, shape=(rows, cols)); landuse_f.from_numpy(land_np)
    output_f  = ti.field(ti.f32, shape=(rows, cols))

    ebf_f = ti.field(ti.f32, shape=ebf_np.shape); ebf_f.from_numpy(ebf_np)
    dbf_f = ti.field(ti.f32, shape=dbf_np.shape); dbf_f.from_numpy(dbf_np)
    enf_f = ti.field(ti.f32, shape=enf_np.shape); enf_f.from_numpy(enf_np)
    dnf_f = ti.field(ti.f32, shape=dnf_np.shape); dnf_f.from_numpy(dnf_np)
    ndvi_lai_f = ti.field(ti.f32, shape=ndvi_lai_np.shape); ndvi_lai_f.from_numpy(ndvi_lai_np)
    print("Computing LAI...")
    # Run
    compute_lai(ebf_np.shape[0], dbf_np.shape[0], enf_np.shape[0], dnf_np.shape[0], ndvi_lai_np.shape[0])

    # Export
    out = output_f.to_numpy()
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_tif, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(out)
    band.SetNoDataValue(-9999.0)
    out_ds.FlushCache()
    print(f"Processing complete -> {output_tif}  (SZA={sza}°)")

# ---------- 6. CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LUT-based LAI retrieval (3-band input)")
    parser.add_argument("-i", "--input",  required=True, help="Input 3-band GeoTIFF (Red,NIR,Landuse)")
    parser.add_argument("-o", "--output", required=True, help="Output LAI GeoTIFF")
    parser.add_argument("-sza", type=float, required=True, help="Solar zenith angle to keep in LUT (must be provided)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    parser.add_argument("--sensor", choices=["L8", "L9"], default="L8",
                    help="Sensor, e.g., Landsat mission: L8 or L9 (default: L8)")
    args = parser.parse_args()
    
    auto_init(force_cpu=args.cpu)

    t0 = time.time()
    run_lookup(args.input, args.output, args.sza, args.sensor)
    print("Elapsed time: %.2f s" % (time.time() - t0))