import os
import open3d as o3d
import teaserpp_python
import numpy as np
import copy
import time
import pandas as pd
import cv2

from helpers import *
# Ensure that the helper functions from 'helpers.py' are available:
# - pcd2xyz
# - extract_fpfh
# - find_correspondences
# - get_teaser_solver
# - Rt2T

def load_intrinsic_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    intrinsic_matrix = []
    for line in lines:
        numbers = [float(num) for num in line.strip().split()]
        intrinsic_matrix.append(numbers)
    intrinsic_matrix = np.array(intrinsic_matrix)
    return intrinsic_matrix

def project_points_to_image_plane(points_3d, intrinsic_matrix, transformation_matrix):
    # Transform points with the transformation matrix
    points_3d_hom = np.hstack(
        (points_3d.T, np.ones((points_3d.shape[1], 1)))
    )
    transformed_points = (
        (transformation_matrix @ points_3d_hom.T).T[:, :3]
    )

    # Project transformed points to 2D
    projected_points = intrinsic_matrix @ transformed_points.T
    projected_points = (projected_points[:2] / projected_points[2]).T

    return projected_points

def overlay_point_cloud_on_image(
    source_pcd, image, transformation_matrix, intrinsic_matrix, output_path
):
    points_3d = pcd2xyz(source_pcd)  # Shape (3, N)

    # Project points to 2D
    projected_points = project_points_to_image_plane(
        points_3d, intrinsic_matrix, transformation_matrix
    )

    # Create an overlay layer
    overlay = image.copy()

    # Draw points on the overlay
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            cv2.circle(
                overlay, (x, y), radius=2, color=(0, 0, 255), thickness=-1
            )

    # Blend the overlay with the original image
    alpha = 0.50
    blended_image = cv2.addWeighted(
        overlay, alpha, image, 1 - alpha, 0
    )

    # Save the blended image
    cv2.imwrite(output_path, blended_image)

def compute_metrics(estimated_matrix, ground_truth_matrix):
    translation_est = estimated_matrix[:3, 3]
    translation_gt = ground_truth_matrix[:3, 3]
    # Convert translation vectors from meters to millimeters
    translation_error_m = np.linalg.norm(translation_est - translation_gt)
    translation_error_mm = translation_error_m * 1000  # Convert to mm

    rotation_est = estimated_matrix[:3, :3]
    rotation_gt = ground_truth_matrix[:3, :3]
    rotation_diff = np.dot(rotation_est, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rotation_error = np.arccos(trace)
    return translation_error_mm, np.degrees(rotation_error)

def add_metric(model_points, R_gt, t_gt, R_pred, t_pred):
    """
    Calculates the Average Distance of Model Points (ADD) metric.
    """
    # Transform points using ground truth
    transformed_gt = (R_gt @ model_points.T).T + t_gt

    # Transform points using predictions
    transformed_pred = (R_pred @ model_points.T).T + t_pred

    # Compute average distance
    add = np.mean(np.linalg.norm(transformed_gt - transformed_pred, axis=1))
    return add

def main():
    VOXEL_SIZE = 0.01
    num_runs = 5  # Number of runs per scene

    # Paths and configurations
    scenes_root_dir = "/home/martyn/Thesis/pose-estimation/data/scenes/"
    output_root_dir = "/home/martyn/Thesis/pose-estimation/results/methods/teaserpp/"
    os.makedirs(output_root_dir, exist_ok=True)

    intrinsic_matrix = load_intrinsic_matrix("/home/martyn/Thesis/pose-estimation/data/cam_K.txt")
    source_path = '/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply'
    source_raw = o3d.io.read_point_cloud(source_path)
    #source_raw.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

    if source_raw.is_empty():
        print("Error: Source point cloud is empty or failed to load.")
        return

    # Prepare lists to collect metrics over all scenes
    all_metrics = []

    for scene_num in range(1, 11):  # Scenes from 1 to 10
        scene_name = f"scene_{scene_num:02d}"
        scene_dir = os.path.join(scenes_root_dir, scene_name)
        output_dir = os.path.join(output_root_dir, scene_name)
        
        os.makedirs(output_dir, exist_ok=True)

        # Load target data
        target_path = os.path.join(scene_dir, "point_cloud_cropped.ply")
        ground_truth_path = os.path.join(scene_dir, "tf_ground_truth.txt")
        rgb_image_path = os.path.join(scene_dir, "rgb.png")

        target_raw = o3d.io.read_point_cloud(target_path)
        #target_raw.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        ground_truth = np.loadtxt(ground_truth_path)
        image = cv2.imread(rgb_image_path)

        # Error handling
        if target_raw.is_empty():
            print(f"Error: Target point cloud is empty or failed to load for {scene_name}.")
            continue
        if ground_truth.size == 0:
            print(f"Error: Ground truth transformation matrix failed to load for {scene_name}.")
            continue
        if image is None:
            print(f"Error: Failed to load image for {scene_name}.")
            continue

        # Initialize lists to store metrics and runtimes for this scene
        descriptor_times = []
        registration_times = []
        icp_times = []
        total_runtimes = []
        teaser_translation_errors = []
        teaser_rotation_errors = []
        icp_translation_errors = []
        icp_rotation_errors = []
        add_metrics = []

        for i in range(num_runs):
            print(f"\n{scene_name} - Run {i+1}/{num_runs}")
            run_dir = os.path.join(output_dir, f"run_{i+1:02d}")
            os.makedirs(run_dir, exist_ok=True)

            total_start_time = time.time()

            # Voxel downsample both clouds for feature extraction
            source_pcd = source_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
            target_pcd = target_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)

            # Descriptor computation (FPFH)
            print("Computing FPFH features...")
            descriptor_start_time = time.time()
            source_feats = extract_fpfh(source_pcd, VOXEL_SIZE)
            target_feats = extract_fpfh(target_pcd, VOXEL_SIZE)
            descriptor_time = time.time() - descriptor_start_time
            descriptor_times.append(descriptor_time)
            print(f"Descriptor computation time: {descriptor_time:.2f} seconds")

            # Establish correspondences
            print("Finding correspondences...")
            source_xyz = pcd2xyz(source_pcd)
            target_xyz = pcd2xyz(target_pcd)
            corrs_source, corrs_target = find_correspondences(
                source_feats, target_feats, mutual_filter=True)
            source_corr = source_xyz[:, corrs_source]
            target_corr = target_xyz[:, corrs_target]
            num_corrs = source_corr.shape[1]
            print(f"Found {num_corrs} correspondences.")

            # TEASER++ Registration
            print("Running TEASER++ registration...")
            registration_start_time = time.time()
            teaser_solver = get_teaser_solver(VOXEL_SIZE * 1.5)
            teaser_solver.solve(source_corr, target_corr)
            solution = teaser_solver.getSolution()
            R_teaser = solution.rotation
            t_teaser = solution.translation
            T_teaser = Rt2T(R_teaser, t_teaser)
            registration_time = time.time() - registration_start_time
            print(f"TEASER++ registration time: {registration_time:.2f} seconds")

            # ICP Refinement using original point clouds
            print("Running ICP refinement with original point clouds...")
            icp_start_time = time.time()
            icp_sol = o3d.pipelines.registration.registration_icp(
                source_raw, target_raw, VOXEL_SIZE * 0.4, T_teaser,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            T_icp = icp_sol.transformation
            icp_time = time.time() - icp_start_time
            print(f"ICP refinement time: {icp_time:.2f} seconds")

            total_runtime = time.time() - total_start_time
            total_runtimes.append(total_runtime)

            # Save transformation matrices
            np.savetxt(os.path.join(run_dir, "teaser_transformation.txt"), T_teaser)
            np.savetxt(os.path.join(run_dir, "icp_transformation.txt"), T_icp)

            # Compute metrics
            teaser_translation_error, teaser_rotation_error = compute_metrics(
                T_teaser, ground_truth
            )
            icp_translation_error, icp_rotation_error = compute_metrics(
                T_icp, ground_truth
            )

            # Compute ADD metrics
            R_pred = T_icp[:3, :3]
            t_pred = T_icp[:3, 3]
            R_gt = ground_truth[:3, :3]
            t_gt = ground_truth[:3, 3]
            add_error = add_metric(np.asarray(source_raw.points), R_gt, t_gt, R_pred, t_pred)

            teaser_translation_errors.append(teaser_translation_error)
            teaser_rotation_errors.append(teaser_rotation_error)
            icp_translation_errors.append(icp_translation_error)
            icp_rotation_errors.append(icp_rotation_error)
            registration_times.append(registration_time)
            icp_times.append(icp_time)
            add_metrics.append(add_error)

            # Save runtime information
            with open(os.path.join(run_dir, "runtime.txt"), "w") as f:
                f.write(f"Total Runtime: {total_runtime:.4f} seconds\n")
                f.write(f"Descriptor Computation Time: {descriptor_time:.4f} seconds\n")
                f.write(f"TEASER++ Registration Time: {registration_time:.4f} seconds\n")
                f.write(f"ICP Refinement Time: {icp_time:.4f} seconds\n")

            # Save metrics
            with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
                f.write(f"TEASER Translation Error: {teaser_translation_error:.6f} mm\n")
                f.write(f"TEASER Rotation Error: {teaser_rotation_error:.6f} deg\n")
                f.write(f"ICP Translation Error: {icp_translation_error:.6f} mm\n")
                f.write(f"ICP Rotation Error: {icp_rotation_error:.6f} deg\n")
                f.write(f"ADD: {add_error:.6f} deg\n")

            # Overlay 3D points on 2D image and save
            print("Creating overlay image...")
            # Apply the transformation to the original source point cloud
            transformed_source_raw = copy.deepcopy(source_raw).transform(T_icp)
            overlay_point_cloud_on_image(
                transformed_source_raw, image, np.identity(4), intrinsic_matrix,
                os.path.join(run_dir, "overlay.png")
            )

            print(f"{scene_name} - Run {i+1} results saved in {run_dir}")

        # Save overall metrics to CSV for this scene
        metrics = {
            "Run": list(range(1, num_runs + 1)),
            "TEASER Translation Error (mm)": teaser_translation_errors,
            "TEASER Rotation Error (deg)": teaser_rotation_errors,
            "ICP Translation Error (mm)": icp_translation_errors,
            "ICP Rotation Error (deg)": icp_rotation_errors,
            "ADD (mm)": add_metrics,
            "Descriptor Computation Time (s)": descriptor_times,
            "TEASER++ Registration Time (s)": registration_times,
            "ICP Refinement Time (s)": icp_times,
            "Total Runtime (s)": total_runtimes
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, "metrics_over_runs.csv"), index=False)
        print(f"{scene_name} - Metrics saved to:", os.path.join(output_dir, "metrics_over_runs.csv"))

        # -------------------------------------------------
        # Compute means (averages) and standard deviations
        # -------------------------------------------------
        avg_total_runtime = np.mean(total_runtimes)
        sd_total_runtime = np.std(total_runtimes)

        avg_desc_time = np.mean(descriptor_times)
        sd_desc_time = np.std(descriptor_times)

        avg_reg_time = np.mean(registration_times)
        sd_reg_time = np.std(registration_times)

        avg_icp_time = np.mean(icp_times)
        sd_icp_time = np.std(icp_times)

        avg_teaser_trans_err = np.mean(teaser_translation_errors)
        sd_teaser_trans_err = np.std(teaser_translation_errors)

        avg_teaser_rot_err = np.mean(teaser_rotation_errors)
        sd_teaser_rot_err = np.std(teaser_rotation_errors)

        avg_icp_trans_err = np.mean(icp_translation_errors)
        sd_icp_trans_err = np.std(icp_translation_errors)

        avg_icp_rot_err = np.mean(icp_rotation_errors)
        sd_icp_rot_err = np.std(icp_rotation_errors)

        avg_add = np.mean(add_metrics)
        sd_add = np.std(add_metrics)

        # Print scene-level summary to console
        print(f"\n{scene_name} - Averages over runs:")
        print(f"Average Total Runtime (s): {avg_total_runtime:.6f}")
        print(f"Average Descriptor Computation Time (s): {avg_desc_time:.6f}")
        print(f"Average Registration Runtime (s): {avg_reg_time:.6f}")
        print(f"Average Refinement Runtime (s): {avg_icp_time:.6f}")
        print(f"Average TEASER Translation Error (mm): {avg_teaser_trans_err:.6f}")
        print(f"Average TEASER Rotation Error (deg): {avg_teaser_rot_err:.6f}")
        print(f"Average ICP Translation Error (mm): {avg_icp_trans_err:.6f}")
        print(f"Average ICP Rotation Error (deg): {avg_icp_rot_err:.6f}")
        print(f"Average ADD (mm): {avg_add:.6f}")

        print(f"\n{scene_name} - Standard Deviations over runs:")
        print(f"Total Runtime (s) SD: {sd_total_runtime:.6f}")
        print(f"Descriptor Computation Time (s) SD: {sd_desc_time:.6f}")
        print(f"Registration Runtime (s) SD: {sd_reg_time:.6f}")
        print(f"Refinement Runtime (s) SD: {sd_icp_time:.6f}")
        print(f"TEASER Translation Error (mm) SD: {sd_teaser_trans_err:.6f}")
        print(f"TEASER Rotation Error (deg) SD: {sd_teaser_rot_err:.6f}")
        print(f"ICP Translation Error (mm) SD: {sd_icp_trans_err:.6f}")
        print(f"ICP Rotation Error (deg) SD: {sd_icp_rot_err:.6f}")
        print(f"ADD (mm) SD: {sd_add:.6f}")

        # -------------------------------------------------
        # Save average + SD metrics to average_metrics.txt
        # -------------------------------------------------
        avg_metrics_file = os.path.join(output_dir, "average_metrics.txt")
        with open(avg_metrics_file, "w") as f:
            f.write(f"Average Metrics over runs for {scene_name}:\n")
            f.write(f"Average Total Runtime (s): {avg_total_runtime:.6f}\n")
            f.write(f"Average Descriptor Computation Time (s): {avg_desc_time:.6f}\n")
            f.write(f"Average Registration Runtime (s): {avg_reg_time:.6f}\n")
            f.write(f"Average Refinement Runtime (s): {avg_icp_time:.6f}\n")
            f.write(f"Average TEASER Translation Error (mm): {avg_teaser_trans_err:.6f}\n")
            f.write(f"Average TEASER Rotation Error (deg): {avg_teaser_rot_err:.6f}\n")
            f.write(f"Average ICP Translation Error (mm): {avg_icp_trans_err:.6f}\n")
            f.write(f"Average ICP Rotation Error (deg): {avg_icp_rot_err:.6f}\n")
            f.write(f"Average ADD (mm): {avg_add:.6f}\n")

            f.write("\nStandard Deviation over runs:\n")
            f.write(f"Total Runtime (s) SD: {sd_total_runtime:.6f}\n")
            f.write(f"Descriptor Computation Time (s) SD: {sd_desc_time:.6f}\n")
            f.write(f"Registration Runtime (s) SD: {sd_reg_time:.6f}\n")
            f.write(f"Refinement Runtime (s) SD: {sd_icp_time:.6f}\n")
            f.write(f"TEASER Translation Error (mm) SD: {sd_teaser_trans_err:.6f}\n")
            f.write(f"TEASER Rotation Error (deg) SD: {sd_teaser_rot_err:.6f}\n")
            f.write(f"ICP Translation Error (mm) SD: {sd_icp_trans_err:.6f}\n")
            f.write(f"ICP Rotation Error (deg) SD: {sd_icp_rot_err:.6f}\n")
            f.write(f"ADD (mm) SD: {sd_add:.6f}\n")

        print(f"{scene_name} - Averages and SD saved to: {avg_metrics_file}")

        # -------------------------------------------------
        # Create a scene-level dictionary including average and SD
        # -------------------------------------------------
        scene_metrics = {
            "Scene": scene_name,
            "Total Runtime (s) Mean": avg_total_runtime,
            "Total Runtime (s) SD": sd_total_runtime,

            "Descriptor Computation Time (s) Mean": avg_desc_time,
            "Descriptor Computation Time (s) SD": sd_desc_time,

            "Registration Runtime (s) Mean": avg_reg_time,
            "Registration Runtime (s) SD": sd_reg_time,

            "Refinement Runtime (s) Mean": avg_icp_time,
            "Refinement Runtime (s) SD": sd_icp_time,

            "Registration Translation Error (mm) Mean": avg_teaser_trans_err,
            "Registration Translation Error (mm) SD": sd_teaser_rot_err,

            "Registration Rotation Error (deg) Mean": avg_teaser_rot_err,
            "Registration Rotation Error (deg) SD": sd_teaser_rot_err,

            "Refinement Translation Error (mm) Mean": avg_icp_trans_err,
            "Refinement Translation Error (mm) SD": sd_icp_trans_err,

            "Refinement Rotation Error (deg) Mean": avg_icp_rot_err,
            "Refinement Rotation Error (deg) SD": sd_icp_rot_err,

            "ADD (mm) Mean": avg_add,
            "ADD (mm) SD": sd_add
        }

        # Append scene metrics (avg + SD) to all_metrics
        all_metrics.append(scene_metrics)

    # After processing all scenes, save all average metrics to a CSV
    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(os.path.join(output_root_dir, "all_scenes_average_metrics.csv"), index=False)
    print("All scenes average metrics saved to:", os.path.join(output_root_dir, "all_scenes_average_metrics.csv"))

if __name__ == "__main__":
    main()
