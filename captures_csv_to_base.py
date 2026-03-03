#!/usr/bin/env python3
import os
import csv
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x,y,z,w) -> rotazione 3x3 (con normalizzazione)."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion con norma ~0 (non valido).")
    qx, qy, qz, qw = (q / n).tolist()

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    return np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=float)


def rot_to_quat(R: np.ndarray):
    """
    Rotazione 3x3 -> quaternion (x,y,z,w)
    """
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def pose_to_T(px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Costruisce ^baseT_ee (cioè T ee->base nel tuo uso)."""
    R = quat_to_rot(qx, qy, qz, qw)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array([px, py, pz], dtype=float)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inversione trasformazione rigida 4x4."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


class CapturesCsvToBase(Node):
    def __init__(self):
        super().__init__("captures_csv_to_base")

        # ===== Parametri =====
        self.declare_parameter(
            "csv_path",
            "/home/elenablaco/lbr_ws/Downloads/validation22dec/captures.csv"
        )
        self.declare_parameter("output_path", "")  # se vuoto => auto out_points.csv nella stessa cartella
        self.declare_parameter("base_frame", "lbr_link_0")
        self.declare_parameter("pose_is_T_base_ee", True)

        self.declare_parameter("publish_markers", True)

        # TF
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("ee_tf_child", "ee_csv")
        self.declare_parameter("cam_tf_child", "camera_csv")
        self.declare_parameter("cycle_tf", True)   # se true, scorre tutte le righe del csv
        self.declare_parameter("cycle_hz", 2.0)    # frequenza di scorrimento

        csv_path = self.get_parameter("csv_path").value
        out_path = self.get_parameter("output_path").value
        self.base_frame = self.get_parameter("base_frame").value
        self.pose_is_T_base_ee = bool(self.get_parameter("pose_is_T_base_ee").value)

        self.publish_markers = bool(self.get_parameter("publish_markers").value)

        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.ee_tf_child = self.get_parameter("ee_tf_child").value
        self.cam_tf_child = self.get_parameter("cam_tf_child").value
        self.cycle_tf = bool(self.get_parameter("cycle_tf").value)
        self.cycle_hz = float(self.get_parameter("cycle_hz").value)

        if not os.path.exists(csv_path):
            raise RuntimeError(f"File non trovato: {csv_path}")

        # output default
        if not out_path:
            out_path = os.path.join(os.path.dirname(csv_path), "out_points.csv")
        self.out_path = out_path

        # ====== T_cam->ee (tua calibrazione):  ^eeT_cam ======
        self.T_cam_to_ee = np.array([
            [0.99997, -0.00314, -0.00725,  0.00858],
            [0.00607,  0.89345,  0.44913, -0.44913],
            [0.01713, -0.44960,  0.89344,  0.02955],
            [0.0,      0.0,      0.0,      1.0],
        ], dtype=float)

        # Marker publisher (latching per RViz)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.marker_pub = self.create_publisher(Marker, "points_in_base", qos)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        # Storage
        self.points_base = []
        self.rows_out = []
        self.T_base_ee_list = []
        self.T_base_cam_list = []

        self.process_csv(csv_path)
        self.write_output(self.out_path)

        self.get_logger().info(f"Output CSV scritto in: {self.out_path}")

        # Marker
        if self.publish_markers and len(self.points_base) > 0:
            self.marker = self.make_marker(self.points_base)
            self.timer_marker = self.create_timer(1.0, self.publish_marker)
            self.get_logger().info(
                f"Pubblico {len(self.points_base)} punti su /points_in_base (frame: {self.base_frame})."
            )

        # TF
        if self.publish_tf and len(self.T_base_ee_list) > 0:
            if self.cycle_tf:
                self.tf_index = 0
                period = max(1e-3, 1.0 / self.cycle_hz)
                self.timer_tf = self.create_timer(period, self.publish_tf_step)
                self.get_logger().info(
                    f"Pubblico TF ciclico: {self.base_frame}->{self.ee_tf_child} e {self.base_frame}->{self.cam_tf_child} @ {self.cycle_hz} Hz"
                )
            else:
                # pubblica sempre la prima riga (statica “di fatto” ma con timestamp aggiornato)
                self.tf_index = 0
                self.timer_tf = self.create_timer(0.05, self.publish_tf_step)
                self.get_logger().info(
                    f"Pubblico TF fisso (prima riga): {self.base_frame}->{self.ee_tf_child} e {self.base_frame}->{self.cam_tf_child}"
                )

    def process_csv(self, csv_path: str):
        expected = [
            "capture_id","image_filename","centroid_txt","cx","cy",
            "Xf","Yf","Zf","px","py","pz","qx","qy","qz","qw","timestamp"
        ]

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter=",")
            if reader.fieldnames != expected:
                self.get_logger().warn(f"Header diverso dall'atteso.\nTrovato: {reader.fieldnames}\nAtteso: {expected}")

            for row in reader:
                # Posa EE
                px = float(row["px"]); py = float(row["py"]); pz = float(row["pz"])
                qx = float(row["qx"]); qy = float(row["qy"]); qz = float(row["qz"]); qw = float(row["qw"])

                T_base_ee = pose_to_T(px, py, pz, qx, qy, qz, qw)
                if not self.pose_is_T_base_ee:
                    T_base_ee = invert_T(T_base_ee)

                # ^baseT_cam = ^baseT_ee * ^eeT_cam
                T_base_cam = T_base_ee @ self.T_cam_to_ee

                # Punto in camera (dal CSV)
                Xf = float(row["Xf"]); Yf = float(row["Yf"]); Zf = float(row["Zf"])
                p_cam = np.array([Xf, Yf, Zf, 1.0], dtype=float)

                # p_base = ^baseT_cam * p_cam
                p_base = (T_base_cam @ p_cam)[:3]
                self.points_base.append(p_base)

                # salva per TF
                self.T_base_ee_list.append(T_base_ee)
                self.T_base_cam_list.append(T_base_cam)

                # salva output file
                self.rows_out.append({
                    "capture_id": row["capture_id"],
                    "image_filename": row["image_filename"],
                    "T_base_ee": T_base_ee.reshape(-1).tolist(),
                    "p_base": p_base.tolist()
                })

        # log controllo
        T0 = self.T_base_ee_list[0]
        self.get_logger().info("Esempio ^baseT_ee (prima riga):\n" + str(T0))

    def write_output(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None

        header = (
            ["capture_id", "image_filename"]
            + [f"T_base_ee_{r}{c}" for r in range(4) for c in range(4)]
            + ["p_base_x", "p_base_y", "p_base_z"]
        )

        with open(out_path, "w", newline="") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(header)
            for r in self.rows_out:
                w.writerow([r["capture_id"], r["image_filename"]] + r["T_base_ee"] + r["p_base"])

    def make_marker(self, points_base):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.ns = "captures_points"
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD

        m.scale.x = 0.01
        m.scale.y = 0.01

        m.color.a = 1.0
        m.color.r = 0.1
        m.color.g = 0.9
        m.color.b = 0.1

        m.points = []
        for p in points_base:
            pt = Point()
            pt.x, pt.y, pt.z = float(p[0]), float(p[1]), float(p[2])
            m.points.append(pt)

        return m

    def publish_marker(self):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.marker)

    def T_to_tf(self, T: np.ndarray, child_frame: str) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_frame
        msg.child_frame_id = child_frame

        msg.transform.translation.x = float(T[0, 3])
        msg.transform.translation.y = float(T[1, 3])
        msg.transform.translation.z = float(T[2, 3])

        qx, qy, qz, qw = rot_to_quat(T[:3, :3])
        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw
        return msg

    def publish_tf_step(self):
        if not self.tf_broadcaster:
            return

        i = self.tf_index
        T_bee = self.T_base_ee_list[i]
        T_bcam = self.T_base_cam_list[i]

        self.tf_broadcaster.sendTransform(self.T_to_tf(T_bee, self.ee_tf_child))
        self.tf_broadcaster.sendTransform(self.T_to_tf(T_bcam, self.cam_tf_child))

        if self.cycle_tf:
            self.tf_index = (self.tf_index + 1) % len(self.T_base_ee_list)


def main():
    rclpy.init()
    node = CapturesCsvToBase()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()