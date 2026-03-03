#!/usr/bin/env python3
"""
point_visualizer_red_vs_blue_mean_match_and_label.py

- Loads RED points from YAML (with GROUP_SIZES averaging + outlier filtering)
- Computes BLUE points from saved images + poses.csv (solvePnP + hand-eye + robot poses)
- Computes mean BLUE per corner across images (12 points)
- Automatically matches BLUE mean points to RED points using minimum-total-distance assignment (Hungarian)
- Prints error stats after matching
- Publishes RViz markers:
    * Red spheres + labels:  ns="yaml_points" + "yaml_labels"   labels "R0..R11"
    * Blue mean spheres + labels: ns="blue_mean" + "blue_labels" labels "B0..B11"
    * Blue mean matched-to-red spheres (cyan) + labels "M0..M11": ns="blue_matched" + "matched_labels"
      where Mi is the blue point matched to red Ri

Topic: /calibration_points_markers
Frame: lbr_link_0

Requires:
  - OpenCV (cv2)
  - numpy, pyyaml
  - scipy (for Hungarian): pip install scipy
"""

import os
import csv
import yaml
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


# ---------- Paths ----------
# ---------- Paths ----------
YAML_FILE = "/media/sf_handeye_validation/calibration_points_22dec.yaml"
IMG_DIR   = "/media/sf_handeye_validation/validation22dec"
POSES_CSV = os.path.join(IMG_DIR, "poses.csv")

# Number of measurements per red point (in order). 12 corners * 3 measurements = 36 entries
GROUP_SIZES = [3,3,3,3, 3,3,3,3, 3,3,3,3]

# ---------- Grid definition ----------
CHESSBOARD_SIZE = (4, 3)   # inner corners (cols, rows)
GRID_SPACING_M  = 0.016    # meters (16 mm)

# ---------- Camera intrinsics ----------
K = np.array([
    [644.82861328125, 0.0,              641.0790405273438],
    [0.0,             644.1883544921875, 374.2107238769531],
    [0.0,             0.0,              1.0]
], dtype=np.float64)

dist = np.array([
    -0.054780781269073486,
     0.06129075214266777,
    -8.948066533776e-05,
     1.699903623375576e-05
    -0.020003778859972954
], dtype=np.float64)

# ---------- Hand-eye ----------
# You stated: "T_EC is camera in end effector" => T_EC = ^E T_C
# ---------- Hand-eye ----------
# T_EC = ^E T_C  (camera -> end effector)  [METERS]
T_EC = np.array([
    [ 0.99996874, -0.00314454, -0.00725433,  0.00858134],
    [ 0.00606781,  0.89344557,  0.44913049, -0.17134914],
    [ 0.00506904, -0.44916047,  0.89343672,  0.02954765],
    [ 0.0,         0.0,         0.0,         1.0]
], dtype=np.float64)


# --------------------------------
# Helpers
# --------------------------------
def filter_outliers(points, z_thresh=2.5):
    """
    points: list of (3,) arrays
    returns: numpy array filtered (M,3)
    """
    arr = np.vstack(points)
    centroid = arr.mean(axis=0)
    dists = np.linalg.norm(arr - centroid, axis=1)
    mean_d = dists.mean()
    std_d  = dists.std()
    if std_d < 1e-9:
        return arr
    z = np.abs((dists - mean_d) / std_d)
    return arr[z < z_thresh]

def quat_to_R(qx, qy, qz, qw):
    # (x,y,z,w) quaternion -> rotation matrix
    x,y,z,w = qx,qy,qz,qw
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx,yy,zz = x*x*s, y*y*s, z*z*s
    xy,xz,yz = x*y*s, x*z*s, y*z*s
    wx,wy,wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1-(yy+zz), xy-wz,     xz+wy],
        [xy+wz,     1-(xx+zz), yz-wx],
        [xz-wy,     yz+wx,     1-(xx+yy)]
    ], dtype=np.float64)

def load_poses(csv_path):
    """
    Returns list of (image_filename, T_base_ee).
    Sniffs delimiter among comma/tab/semicolon.
    """
    poses = []
    with open(csv_path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            img = row["image_filename"].strip()
            px,py,pz = float(row["px"]), float(row["py"]), float(row["pz"])
            qx,qy,qz,qw = float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])
            R = quat_to_R(qx,qy,qz,qw)
            T = make_T(R, np.array([px,py,pz]))
            poses.append((img, T))
    return poses

def grid_object_points():
    cols, rows = CHESSBOARD_SIZE
    obj = []
    for r in range(rows):
        for c in range(cols):
            obj.append([c*GRID_SPACING_M, r*GRID_SPACING_M, 0.0])
    return np.array(obj, dtype=np.float64)  # (N,3)

def error_report(red_pts, blue_pts, logger):
    red  = np.asarray(red_pts, dtype=np.float64)
    blue = np.asarray(blue_pts, dtype=np.float64)
    diff = blue - red
    dist_e = np.linalg.norm(diff, axis=1)

    rmse = np.sqrt(np.mean(dist_e**2))
    mean = np.mean(dist_e)
    med  = np.median(dist_e)
    mx   = np.max(dist_e)
    std  = np.std(dist_e)

    dx,dy,dz = diff[:,0], diff[:,1], diff[:,2]

    logger.info(f"N points: {len(dist_e)}")
    logger.info(f"Mean |e|   : {mean*1000:.2f} mm")
    logger.info(f"Median |e| : {med*1000:.2f} mm")
    logger.info(f"RMSE |e|   : {rmse*1000:.2f} mm")
    logger.info(f"Std |e|    : {std*1000:.2f} mm")
    logger.info(f"Max |e|    : {mx*1000:.2f} mm")
    logger.info("Axis error (blue - red):")
    logger.info(f"  mean dx = {np.mean(dx)*1000:.2f} mm   std dx = {np.std(dx)*1000:.2f} mm")
    logger.info(f"  mean dy = {np.mean(dy)*1000:.2f} mm   std dy = {np.std(dy)*1000:.2f} mm")
    logger.info(f"  mean dz = {np.mean(dz)*1000:.2f} mm   std dz = {np.std(dz)*1000:.2f} mm")
    logger.info("Per-corner |e| (mm): " + ", ".join([f"{d*1000:.1f}" for d in dist_e]))

def match_by_min_total_distance(red_pts, blue_pts):
    """
    Hungarian assignment: finds one-to-one matching minimizing total distances.
    Returns:
      perm: list where perm[i_red] = j_blue
      dists: per-red matched distance (meters)
    """
    from scipy.optimize import linear_sum_assignment

    red = np.asarray(red_pts, dtype=np.float64)
    blue = np.asarray(blue_pts, dtype=np.float64)
    N = red.shape[0]
    cost = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        cost[i, :] = np.linalg.norm(blue - red[i], axis=1)
    r_ind, b_ind = linear_sum_assignment(cost)

    perm = np.zeros(N, dtype=int)
    perm[r_ind] = b_ind
    dists = cost[np.arange(N), perm]
    return perm.tolist(), dists

def make_text_marker(frame_id, ns, mid, pos, text,
                     scale=0.02, color=(1.0,1.0,1.0,1.0), z_offset=0.01):
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = mid
    m.type = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2] + z_offset)
    m.pose.orientation.w = 1.0
    m.scale.z = float(scale)
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.text = str(text)
    return m


# --------------------------------
# Node
# --------------------------------
class PointVisualizer(Node):
    def __init__(self):
        super().__init__('point_visualizer')

        self.marker_pub = self.create_publisher(MarkerArray, '/calibration_points_markers', 10)

        # 1) RED means
        red_mean = self.load_red_means()
        self.get_logger().info(f"Loaded red mean points: {len(red_mean)}")

        # 2) BLUE mean (12 pts) from images
        blue_mean, blue_stacked = self.compute_blue_mean_from_images()
        if blue_mean is None:
            self.get_logger().error("No valid images -> blue_mean is None. Publishing only red.")
            blue_mean = np.zeros_like(red_mean)

        # 3) Match blue->red by closest assignment (Hungarian)
        if len(blue_mean) == len(red_mean):
            perm, dists = match_by_min_total_distance(red_mean, blue_mean)
            blue_matched = np.array([blue_mean[j] for j in perm], dtype=np.float64)

            self.get_logger().info("Matching (red_index -> blue_index): " + str(list(enumerate(perm))))
            self.get_logger().info("Matched distances (mm): " + ", ".join([f"{d*1000:.1f}" for d in dists]))

            self.get_logger().info("Error after matching (blue_matched - red):")
            error_report(red_mean, blue_matched, self.get_logger())
        else:
            blue_matched = blue_mean
            self.get_logger().warn("Red and blue sizes differ; skipping matching + error stats.")

        # 4) Publish markers (red, blue_mean, blue_matched) + labels
        self.marker_array = self.build_markers(red_mean, blue_mean, blue_matched)
        self.timer = self.create_timer(1.0, self.publish_markers)

    def load_red_means(self):
        with open(YAML_FILE, 'r') as f:
            raw_points = yaml.safe_load(f)

        total_expected = sum(GROUP_SIZES)
        if len(raw_points) != total_expected:
            self.get_logger().warn(f"YAML has {len(raw_points)} entries but expected {total_expected}")

        # group + outlier filter + mean
        grouped = []
        idx = 0
        for size in GROUP_SIZES:
            group = raw_points[idx:idx+size]
            idx += size
            pts = [np.array(entry["position"], dtype=np.float64) for entry in group]
            grouped.append(pts)

        red_mean = []
        for i, pts in enumerate(grouped):
            filtered = filter_outliers(pts)
            if len(filtered) < len(pts):
                self.get_logger().info(f"Red corner {i}: removed {len(pts)-len(filtered)} outliers")
            red_mean.append(filtered.mean(axis=0))

        return np.array(red_mean, dtype=np.float64)

    def compute_blue_mean_from_images(self):
        objp = grid_object_points()
        N = objp.shape[0]
        poses = load_poses(POSES_CSV)

        per_image_base = []

        for (img_name, T_base_ee) in poses:
            img_path = os.path.join(IMG_DIR, img_name)
            if not os.path.exists(img_path):
                self.get_logger().warn(f"Missing image: {img_path}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.get_logger().warn(f"Failed to read: {img_path}")
                continue

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, flags)
            if not ok:
                self.get_logger().warn(f"Chessboard not found in {img_name}")
                continue

            corners = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            ).reshape(-1, 2)

            ok, rvec, tvec = cv2.solvePnP(
                objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                self.get_logger().warn(f"solvePnP failed for {img_name}")
                continue

            R_cam_board, _ = cv2.Rodrigues(rvec)
            t_cam_board = tvec.reshape(3)

            # points in camera frame: p_C = R*p_O + t
            points_cam = (R_cam_board @ objp.T).T + t_cam_board.reshape(1, 3)  # (N,3)

            # Transform to base like realtime script:
            # T_base_cam = T_base_ee @ T_EC (T_EC is ^E T_C)
            T_base_cam = T_base_ee @ T_EC

            points_base = np.zeros((N, 3), dtype=np.float64)
            for i in range(N):
                p = points_cam[i]
                ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
                pb = (T_base_cam @ ph)[:3]
                points_base[i, :] = pb

            per_image_base.append(points_base)
            self.get_logger().info(f"{img_name}: accepted ({N} corners)")

        if len(per_image_base) == 0:
            return None, None

        stacked = np.stack(per_image_base, axis=0)  # (M,N,3)
        blue_mean = stacked.mean(axis=0)            # (N,3)

        self.get_logger().info(f"Computed blue mean from {stacked.shape[0]} images.")
        return blue_mean, stacked

    def build_markers(self, red_mean, blue_mean, blue_matched):
        ma = MarkerArray()

        # --- RED
        for i, pos in enumerate(red_mean):
            m = Marker()
            m.header.frame_id = "lbr_link_0"
            m.ns = "yaml_points"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.006
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.2
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            ma.markers.append(m)

            
            

        # --- BLUE MEAN (original index from OpenCV order)
        base_id_offset = 1000
        for i, pos in enumerate(blue_mean):
            m = Marker()
            m.header.frame_id = "lbr_link_0"
            m.ns = "blue_mean"
            m.id = base_id_offset + i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.006
            m.color.a = 0.9
            m.color.r, m.color.g, m.color.b = 0.2, 0.9, 0.2  # green
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            ma.markers.append(m)
        return ma

    def publish_markers(self):
        self.marker_pub.publish(self.marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PointVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
