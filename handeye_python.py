import os
import numpy as np
import cv2
import pandas as pd

# =========================
# PATHS
# =========================
OUT_DIR = "calibration_python"

INTRINSICS_YAML = os.path.join(OUT_DIR, "intrinsics.yaml")
EXTRINSICS_CSV  = os.path.join(OUT_DIR, "extrinsics_per_image_filtered.csv")
POSES_ODS       = "poses.ods"

HAND_EYE_YAML = os.path.join(OUT_DIR, "handeye_Tcamee_meters.yaml")
HAND_EYE_NPZ  = os.path.join(OUT_DIR, "handeye_Tcamee_meters.npz")

HAND_EYE_METHOD = cv2.CALIB_HAND_EYE_TSAI


# =========================
# UTILS
# =========================
def quat_to_R(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def rvec_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return R

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def save_yaml(path, T_cam_ee, T_ee_cam):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("units", "meters")
    fs.write("T_cam_ee", T_cam_ee)
    fs.write("T_ee_cam", T_ee_cam)
    fs.release()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # --- Load extrinsics (camera <- checkerboard) ---
    extr = pd.read_csv(EXTRINSICS_CSV)
    extr["image"] = extr["image"].apply(os.path.basename)

    # Convert camera extrinsics from mm -> meters
    extr["tvec_x_m"] = extr["tvec_x_mm"] * 1e-3
    extr["tvec_y_m"] = extr["tvec_y_mm"] * 1e-3
    extr["tvec_z_m"] = extr["tvec_z_mm"] * 1e-3

    # --- Load robot poses (already in meters) ---
    poses = pd.read_excel(POSES_ODS, engine="odf")
    poses["image"] = poses["image_filename"].apply(os.path.basename)

    # --- Merge ---
    data = pd.merge(extr, poses, on="image", how="inner")
    if len(data) < 5:
        raise RuntimeError("Troppe poche pose corrispondenti per hand-eye.")

    print(f"[OK] Coppie immagine-posa: {len(data)}")

    # --- Prepare OpenCV input ---
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam   = []
    t_target2cam   = []

    for _, r in data.iterrows():
        # Robot: gripper -> base
        Rg = quat_to_R(r.qx, r.qy, r.qz, r.qw)
        tg = np.array([r.px, r.py, r.pz]).reshape(3, 1)   # meters

        # Camera: target -> camera
        Rt = rvec_to_R([r.rvec_x, r.rvec_y, r.rvec_z])
        tt = np.array([r.tvec_x_m, r.tvec_y_m, r.tvec_z_m]).reshape(3, 1)

        R_gripper2base.append(Rg)
        t_gripper2base.append(tg)
        R_target2cam.append(Rt)
        t_target2cam.append(tt)

    # --- Hand-Eye Calibration ---
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=HAND_EYE_METHOD
    )

    T_cam_ee = make_T(R_cam2gripper, t_cam2gripper)   # METERS
    T_ee_cam = invert_T(T_cam_ee)

    print("\n=== HAND-EYE (METERS) ===")
    print("Tcamee (camera -> EE) =\n", T_cam_ee)
    print("\nTeeCam (EE -> camera) =\n", T_ee_cam)

    # --- Save ---
    save_yaml(HAND_EYE_YAML, T_cam_ee, T_ee_cam)

    np.savez(
        HAND_EYE_NPZ,
        units="meters",
        T_cam_ee=T_cam_ee,
        T_ee_cam=T_ee_cam,
        R_cam2gripper=R_cam2gripper,
        t_cam2gripper=t_cam2gripper
    )

    print("\nSalvati:")
    print(f"- {HAND_EYE_YAML}")
    print(f"- {HAND_EYE_NPZ}")
