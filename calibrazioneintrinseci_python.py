import os
import glob
import numpy as np
import cv2
import pandas as pd

# =========================
# PATH / INPUT
# =========================
IMAGES_DIR = "images"
POSES_PATH = "poses.ods"

# Checkerboard: INNER CORNERS (cols, rows)
PATTERN_SIZE = (10, 7)      
SQUARE_SIZE_MM = 24.0

# =========================
# OUTPUT / DEBUG
# =========================
OUT_DIR = "calibration_python"
DEBUG_DETECT_DIR = os.path.join(OUT_DIR, "debug_detect")
DEBUG_FAIL_DIR   = os.path.join(OUT_DIR, "debug_fail")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DETECT_DIR, exist_ok=True)
os.makedirs(DEBUG_FAIL_DIR, exist_ok=True)

# Soglia errore riproiezione per scartare immagini
ERROR_THRESH_PX = 0.5

SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


# =========================
# UTILS
# =========================
def read_poses_ods(path: str) -> pd.DataFrame:
    # Richiede: pip install odfpy
    return pd.read_excel(path, engine="odf")

def preprocess(gray):
    # migliora contrasto + riduce rumore
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g

def detect_chessboard(gray, pattern_size):
    """
    Tenta prima findChessboardCornersSB (robusto), poi fallback classico.
    Ritorna (ok, corners) con corners shape (N,1,2).
    """
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(
            gray, pattern_size,
            flags=(cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        )
        if ok:
            return True, corners

    ok, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=(cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    )
    if not ok:
        return False, None

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRIT)
    return True, corners

def make_object_points(pattern_size, square_size_mm):
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp

def reprojection_rmse(objpoints_list, imgpoints_list, rvecs, tvecs, K, dist):
    total_err2 = 0.0
    total_pts = 0
    for i in range(len(objpoints_list)):
        proj, _ = cv2.projectPoints(objpoints_list[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints_list[i], proj, cv2.NORM_L2)
        n = len(objpoints_list[i])
        total_err2 += (err * err)
        total_pts += n
    return float(np.sqrt(total_err2 / total_pts))

def per_image_reproj_error_px(objp, corners, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)  # (N,1,2)
    diff = corners.astype(np.float64) - proj.astype(np.float64)
    err2 = (diff[:, 0, 0]**2 + diff[:, 0, 1]**2)
    return float(np.sqrt(np.mean(err2)))

def draw_like_example(img_bgr, pattern_size, corners, K, dist, rvec, tvec, square_size_mm):
    vis = img_bgr.copy()

    # Corner rilevati
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)

    # Origin = primo corner
    origin_px = tuple(np.round(corners[0, 0]).astype(int))
    cv2.rectangle(vis,
                  (origin_px[0] - 6, origin_px[1] - 6),
                  (origin_px[0] + 6, origin_px[1] + 6),
                  (0, 165, 255), 2)
    cv2.putText(vis, "(0,0)", (origin_px[0] + 10, origin_px[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, "(0,0)", (origin_px[0] + 10, origin_px[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    # Assi (lunghezza 3 quadretti)
    L = 3.0 * square_size_mm
    axis3d = np.float32([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, -L]])
    axis2d, _ = cv2.projectPoints(axis3d, rvec, tvec, K, dist)
    axis2d = axis2d.reshape(-1, 2)

    o = tuple(np.round(axis2d[0]).astype(int))
    x = tuple(np.round(axis2d[1]).astype(int))
    y = tuple(np.round(axis2d[2]).astype(int))
    z = tuple(np.round(axis2d[3]).astype(int))

    cv2.line(vis, o, x, (0, 0, 255), 3)   # X rosso
    cv2.line(vis, o, y, (0, 255, 0), 3)   # Y verde
    cv2.line(vis, o, z, (255, 0, 0), 3)   # Z blu

    cv2.putText(vis, "X", (x[0] + 8, x[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, "Y", (y[0] + 8, y[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, "Z", (z[0] + 8, z[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3, cv2.LINE_AA)

    # Punti riproiettati (croci rosse)
    objp_all = make_object_points(pattern_size, square_size_mm)
    proj, _ = cv2.projectPoints(objp_all, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    for p in proj:
        u, v = np.round(p).astype(int)
        cv2.drawMarker(vis, (u, v), (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
    return vis


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Carica poses
    df = read_poses_ods(POSES_PATH)
    if "image_filename" not in df.columns:
        raise RuntimeError("poses.ods deve contenere la colonna 'image_filename'.")

    # Costruisci lista immagini dalla tabella (corrispondenza garantita)
    image_files = []
    for name in df["image_filename"].astype(str).tolist():
        path = os.path.join(IMAGES_DIR, name)
        if not os.path.exists(path):
            matches = glob.glob(os.path.join(IMAGES_DIR, os.path.basename(name)))
            if matches:
                path = matches[0]
        if not os.path.exists(path):
            print(f"[WARN] Immagine non trovata: {name}")
            continue
        image_files.append(path)

    if not image_files:
        raise RuntimeError("Nessuna immagine trovata (controlla IMAGES_DIR e image_filename in poses.ods).")

    print(f"Trovate {len(image_files)} immagini referenziate da poses.ods")
    print(f"Pattern fisso INNER corners = {PATTERN_SIZE[0]}x{PATTERN_SIZE[1]}")

    # --- 1) DETECTION + raccolta punti per calibrazione ---
    objpoints = []
    imgpoints = []
    img_size = None
    used_files = []

    objp_ref = make_object_points(PATTERN_SIZE, SQUARE_SIZE_MM)

    for i, f in enumerate(image_files):
        img = cv2.imread(f)
        if img is None:
            print(f"[WARN] Immagine non leggibile: {f}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w,h)

        g = preprocess(gray)

        ok, corners = detect_chessboard(g, PATTERN_SIZE)
        if not ok:
            print(f"[MISS] {os.path.basename(f)}")
            if i < 15:
                cv2.imwrite(os.path.join(DEBUG_FAIL_DIR, os.path.basename(f)), g)
            continue

        objpoints.append(objp_ref)
        imgpoints.append(corners)
        used_files.append(f)

        print(f"[OK]  {os.path.basename(f)}")

    if len(objpoints) < 8:
        raise RuntimeError(
            f"Troppe poche immagini valide: {len(objpoints)}.\n"
            f"Guarda {DEBUG_FAIL_DIR}/ e verifica scacchiera intera, fuoco, riflessi."
        )

    # --- 2) Calibrazione intrinseci (prima passata) ---
    ret, K, dist, rvecs_cal, tvecs_cal = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    rmse = reprojection_rmse(objpoints, imgpoints, rvecs_cal, tvecs_cal, K, dist)

    print("\n=== INTRINSECI (prima calibrazione) ===")
    print("K=\n", K)
    print("dist=", dist.ravel())
    print("OpenCV RMS =", ret)
    print("Reproj RMSE (px) =", rmse)

    # Salva intrinseci (prima passata)
    fs = cv2.FileStorage(os.path.join(OUT_DIR, "intrinsics_initial.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("image_width", int(img_size[0]))
    fs.write("image_height", int(img_size[1]))
    fs.write("pattern_cols_inner", int(PATTERN_SIZE[0]))
    fs.write("pattern_rows_inner", int(PATTERN_SIZE[1]))
    fs.write("square_size_mm", float(SQUARE_SIZE_MM))
    fs.write("filter_reproj_threshold_px", float(ERROR_THRESH_PX))
    fs.release()

    # --- 3) Extrinseci + filtro errore + debug ---
    extrinsics_rows = []
    kept_rows = []
    rejected_rows = []

    for idx, f in enumerate(used_files):
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = preprocess(gray)

        ok, corners = detect_chessboard(g, PATTERN_SIZE)
        if not ok:
            continue

        ok_pnp, rvec, tvec = cv2.solvePnP(
            objp_ref, corners, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok_pnp:
            continue

        err_px = per_image_reproj_error_px(objp_ref, corners, rvec, tvec, K, dist)

        row = {
            "image": os.path.basename(f),
            "reproj_rmse_px": float(err_px),
            "rvec_x": float(rvec[0]), "rvec_y": float(rvec[1]), "rvec_z": float(rvec[2]),
            "tvec_x_mm": float(tvec[0]), "tvec_y_mm": float(tvec[1]), "tvec_z_mm": float(tvec[2]),
        }

        dbg = draw_like_example(img, PATTERN_SIZE, corners, K, dist, rvec, tvec, SQUARE_SIZE_MM)

        if err_px > ERROR_THRESH_PX:
            rejected_rows.append(row)
            print(f"[REJECT] {os.path.basename(f)}  err={err_px:.3f}px")
            out_dbg = os.path.join(DEBUG_DETECT_DIR, f"{os.path.basename(f)}_REJECT_err{err_px:.3f}.png")
            cv2.imwrite(out_dbg, dbg)
            continue

        extrinsics_rows.append(row)
        kept_rows.append({"image": os.path.basename(f), "reproj_rmse_px": float(err_px)})
        print(f"[KEEP]   {os.path.basename(f)}  err={err_px:.3f}px")
        out_dbg = os.path.join(DEBUG_DETECT_DIR, f"{os.path.basename(f)}_KEEP_err{err_px:.3f}.png")
        cv2.imwrite(out_dbg, dbg)

    print(f"\nImmagini KEEP: {len(kept_rows)}")
    print(f"Immagini REJECT (err > {ERROR_THRESH_PX}px): {len(rejected_rows)}")

    if len(extrinsics_rows) < 8:
        raise RuntimeError(
            "Dopo il filtro rimangono troppe poche immagini per una calibrazione stabile.\n"
            "Prova ad aumentare ERROR_THRESH_PX o controlla le immagini REJECT."
        )

    # Salva liste + extrinseci filtrati
    pd.DataFrame(kept_rows).to_csv(os.path.join(OUT_DIR, "kept_images.csv"), index=False)
    pd.DataFrame(rejected_rows).to_csv(os.path.join(OUT_DIR, "rejected_images.csv"), index=False)
    pd.DataFrame(extrinsics_rows).to_csv(os.path.join(OUT_DIR, "extrinsics_per_image_filtered.csv"), index=False)

    # --- 4) Ricalibra intrinseci usando SOLO immagini KEEP ---
    name2file = {os.path.basename(p): p for p in used_files}
    kept_names = [r["image"] for r in extrinsics_rows]

    objpoints_f = []
    imgpoints_f = []
    for name in kept_names:
        f = name2file[name]
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = preprocess(gray)

        ok, corners = detect_chessboard(g, PATTERN_SIZE)
        if not ok:
            continue

        objpoints_f.append(objp_ref)
        imgpoints_f.append(corners)

    ret2, K2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints_f, imgpoints_f, img_size, None, None)
    rmse2 = reprojection_rmse(objpoints_f, imgpoints_f, rvecs2, tvecs2, K2, dist2)

    print("\n=== INTRINSECI (ricalibrati su KEEP) ===")
    print("K=\n", K2)
    print("dist=", dist2.ravel())
    print("OpenCV RMS =", ret2)
    print("Reproj RMSE (px) =", rmse2)

    # Salva intrinseci finali
    fs = cv2.FileStorage(os.path.join(OUT_DIR, "intrinsics.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("K", K2)
    fs.write("dist", dist2)
    fs.write("image_width", int(img_size[0]))
    fs.write("image_height", int(img_size[1]))
    fs.write("pattern_cols_inner", int(PATTERN_SIZE[0]))
    fs.write("pattern_rows_inner", int(PATTERN_SIZE[1]))
    fs.write("square_size_mm", float(SQUARE_SIZE_MM))
    fs.write("filter_reproj_threshold_px", float(ERROR_THRESH_PX))
    fs.write("num_keep_images", int(len(objpoints_f)))
    fs.release()

    # Salva tutto in npz
    np.savez(
        os.path.join(OUT_DIR, "calibration_all.npz"),
        K=K2, dist=dist2, img_size=np.array(img_size),
        pattern_used=np.array(PATTERN_SIZE),
        square_size_mm=float(SQUARE_SIZE_MM),
        filter_reproj_threshold_px=float(ERROR_THRESH_PX),
        num_keep_images=int(len(objpoints_f))
    )

    print(f"\nSalvati in {OUT_DIR}/:")
    print(f"- intrinsics_initial.yaml")
    print(f"- intrinsics.yaml (finale dopo filtro)")
    print(f"- extrinsics_per_image_filtered.csv")
    print(f"- kept_images.csv / rejected_images.csv")
    print(f"- calibration_all.npz")
    print(f"Debug immagini con assi in: {DEBUG_DETECT_DIR}/")
