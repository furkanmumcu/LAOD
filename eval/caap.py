import numpy as np

def calculate_iou(box1, box2):
	"""
	Calculates the Intersection over Union (IoU) of two bounding boxes.
	Boxes are expected in format [x_min, y_min, x_max, y_max].
	"""
	x_min_inter = max(box1[0], box2[0])
	y_min_inter = max(box1[1], box2[1])
	x_max_inter = min(box1[2], box2[2])
	y_max_inter = min(box1[3], box2[3])

	inter_width = max(0, x_max_inter - x_min_inter)
	inter_height = max(0, y_max_inter - y_min_inter)
	inter_area = inter_width * inter_height

	box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
	box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

	union_area = box1_area + box2_area - inter_area

	if union_area == 0:
		return 0.0

	return inter_area / union_area


def compute_ap_single_image(predictions_per_image, ground_truths_per_image, iou_threshold):
	"""
	Computes Average Precision (AP) for a single image at a single IoU threshold in a class-agnostic manner.

	Args:
		predictions_per_image (list of list): Predicted detections for one image,
											 format: [x1, y1, x2, y2, confidence].
		ground_truths_per_image (list of list): Ground truths for one image,
												format: [x1, y1, x2, y2, class_category].
												Only box coordinates are used for CA-AP.
		iou_threshold (float): The IoU threshold for considering a detection valid.

	Returns:
		tuple: (list of int: True/False positives, list of int: False negatives mask)
	"""
	num_gt = len(ground_truths_per_image)
	num_pred = len(predictions_per_image)

	if num_pred == 0 and num_gt == 0:
		return [], [], 0  # No detections, no GTs, contributes neutrally to AP calculation

	if num_pred == 0 and num_gt > 0:
		# All ground truths are false negatives if there are no predictions
		return [], [], num_gt  # FP list empty, TP list empty, total FNs = num_gt

	# Sort predictions by confidence in descending order
	predictions_per_image = sorted(predictions_per_image, key=lambda x: x[4], reverse=True)

	true_positives = np.zeros(num_pred)
	false_positives = np.zeros(num_pred)

	gt_matched = np.zeros(num_gt, dtype=bool)

	for i, pred_data in enumerate(predictions_per_image):
		pred_box = pred_data[:4]  # Extract box coordinates

		best_iou = 0.0
		best_gt_idx = -1

		for gt_idx, gt_data in enumerate(ground_truths_per_image):
			gt_box = gt_data[:4]  # Extract box coordinates

			if not gt_matched[gt_idx]:
				iou = calculate_iou(pred_box, gt_box)
				if iou > best_iou:
					best_iou = iou
					best_gt_idx = gt_idx

		if best_gt_idx != -1 and best_iou >= iou_threshold:
			true_positives[i] = 1
			gt_matched[best_gt_idx] = True
		else:
			false_positives[i] = 1

	return true_positives, false_positives, num_gt  # Return counts for cumulative calculation


def compute_ca_ap(all_predictions, all_ground_truths, iou_thresholds):
	"""
	Computes Class-Agnostic Average Precision (CA-AP) across multiple images and IoU thresholds.

	Args:
		all_predictions (list of list of list): List where each element corresponds to an image.
											  predictions[i] = list of detections for image i.
											  Detection format: [x1, y1, x2, y2, confidence].
		all_ground_truths (list of list of list): List where each element corresponds to an image.
												 ground_truths[i] = list of GTs for image i.
												 GT format: [x1, y1, x2, y2, class_category].
		iou_thresholds (list of float): A list of IoU thresholds (e.g., [0.5, 0.55, ..., 0.95]).

	Returns:
		float: The mean Class-Agnostic Average Precision (CA-AP) over all images and specified IoU thresholds.
	"""

	overall_aps = []

	for iou_thresh in iou_thresholds:
		all_tps = []
		all_fps = []
		total_num_gt = 0

		# Collect TPs, FPs, and GT counts for all images at the current IoU threshold
		for img_idx in range(len(all_predictions)):
			predictions_img = all_predictions[img_idx]
			ground_truths_img = all_ground_truths[img_idx]

			tps_img, fps_img, num_gt_img = compute_ap_single_image(
				predictions_img, ground_truths_img, iou_thresh
			)

			all_tps.extend(tps_img)
			all_fps.extend(fps_img)
			total_num_gt += num_gt_img

		# If there are no ground truths at all across all images for this IoU threshold
		if total_num_gt == 0:
			# If there are also no predictions, AP is 1.0 (vacuously true)
			if sum(all_tps) == 0 and sum(all_fps) == 0:
				overall_aps.append(1.0)
			# If there are predictions but no ground truths, all predictions are FPs, AP is 0.0
			else:
				overall_aps.append(0.0)
			continue  # Move to next IoU threshold

		# Combine all predictions (from all images) into one list, retaining original order
		# Need to re-sort all predictions combined to get global confidence ranking
		all_preds_flat = []
		for img_preds in all_predictions:
			all_preds_flat.extend(img_preds)

		# Sort ALL predictions from ALL images by confidence for global PR curve
		all_preds_flat = sorted(all_preds_flat, key=lambda x: x[4], reverse=True)

		# Re-calculate TPs/FPs based on the global confidence ranking
		# This is essentially re-running compute_single_iou_threshold_ap's core logic
		# but with consolidated data

		cumulative_tp = np.zeros(len(all_preds_flat))
		cumulative_fp = np.zeros(len(all_preds_flat))

		# Need to re-do the matching process globally for the AP calculation to be correct
		# This is why standard mAP implementations are complex when doing across multiple images
		# For simplicity and correctness with the new multi-image structure,
		# we will use a common approach: flatten all predictions, assign TPs/FPs
		# by checking against all *unmatched* GTs, then compute global PR.

		# A more robust way: collect matched_pairs and scores, then compute global AP
		matched_pairs_per_iou = []  # (confidence, is_tp_flag) for all predictions
		all_gt_boxes_flat = [gt[:4] for img_gts in all_ground_truths for gt in img_gts]
		num_global_gt = len(all_gt_boxes_flat)

		# This logic ensures that each GT is matched at most once across all predictions from all images
		gt_global_matched = np.zeros(num_global_gt, dtype=bool)

		# Map original GTs to their global index
		global_gt_map = []
		current_global_idx = 0
		for img_gts in all_ground_truths:
			for _ in img_gts:
				global_gt_map.append(current_global_idx)
				current_global_idx += 1

		# Re-iterate through all predictions, sorted globally by confidence
		for pred_data in all_preds_flat:
			pred_box = pred_data[:4]
			pred_confidence = pred_data[4]

			best_iou = 0.0
			best_global_gt_idx = -1

			for global_gt_idx, gt_box_flat in enumerate(all_gt_boxes_flat):
				if not gt_global_matched[global_gt_idx]:
					iou = calculate_iou(pred_box, gt_box_flat)
					if iou > best_iou:
						best_iou = iou
						best_global_gt_idx = global_gt_idx

			is_tp = 0
			if best_global_gt_idx != -1 and best_iou >= iou_thresh:
				is_tp = 1
				gt_global_matched[best_global_gt_idx] = True  # Mark as matched globally

			matched_pairs_per_iou.append((pred_confidence, is_tp))

		# Sort these matched pairs by confidence (already done if we sorted all_preds_flat first)
		# Ensure they are sorted for cumulative sums
		matched_pairs_per_iou = sorted(matched_pairs_per_iou, key=lambda x: x[0], reverse=True)

		if not matched_pairs_per_iou and num_global_gt == 0:
			overall_aps.append(1.0)
			continue
		elif not matched_pairs_per_iou and num_global_gt > 0:
			overall_aps.append(0.0)
			continue

		tps_cumulative = np.cumsum([m[1] for m in matched_pairs_per_iou])
		fps_cumulative = np.cumsum([1 - m[1] for m in matched_pairs_per_iou])  # 1 - is_tp_flag gives fp if not tp

		# Calculate precision and recall
		precision = tps_cumulative / (tps_cumulative + fps_cumulative)
		recall = tps_cumulative / num_global_gt if num_global_gt > 0 else np.zeros_like(tps_cumulative)

		# Interpolate precision (ensure it's non-increasing)
		interpolated_precision = np.maximum.accumulate(precision[::-1])[::-1]

		# Calculate AP (area under PR curve)
		ap = np.sum(
			interpolated_precision * np.diff(np.concatenate(([0.0], recall))))  # Add 0.0 to recall start for diff

		overall_aps.append(ap)

	return np.mean(overall_aps)
