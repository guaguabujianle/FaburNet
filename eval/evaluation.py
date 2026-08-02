import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import SimpleITK
import numpy as np
from eval.ellipse import aop_geometry, drawline_AOD
try:
    from eval.icc import icc, pearson
except ImportError as exc:
    ICC_IMPORT_ERROR = exc
    icc = None
    pearson = None
else:
    ICC_IMPORT_ERROR = None
from tqdm import tqdm


class Evaluator:
    def __init__(self, preds_dir, truth_dir, save_path):
        self.pred_path = Path(preds_dir)
        self.truth_path = Path(truth_dir)
        self.save_path = Path(save_path)
        if not self.pred_path.exists():
            raise FileNotFoundError(f"Prediction directory not found: {self.pred_path}")
        if not self.truth_path.exists():
            raise FileNotFoundError(f"Truth directory not found: {self.truth_path}")
        self.results = {}
        self.results['each_case'] = {}

    def load_image(self, image_path) -> SimpleITK.Image:
        image = SimpleITK.ReadImage(str(image_path))
        return image

    def evaluation(self, pred: SimpleITK.Image, label: SimpleITK.Image):
        pred_data = SimpleITK.GetArrayFromImage(pred).astype(np.uint8)
        label_data = SimpleITK.GetArrayFromImage(label).astype(np.uint8)
        pred_aop_info = self.cal_aop_info(pred_data)
        label_aop_info = self.cal_aop_info(label_data)
        aop_valid = pred_aop_info["valid"] and label_aop_info["valid"]
        aop = abs(pred_aop_info["aop"] - label_aop_info["aop"]) if aop_valid else None

        result = dict()
        result['TP'] = 0
        result['TN'] = 0
        result['FN'] = 0
        result['FP'] = 0
        if aop_valid:
            pred_aop = pred_aop_info["aop"]
            label_aop = label_aop_info["aop"]
            if label_aop >= 120 and pred_aop >= 120:
                result['TP'] = 1
            elif label_aop >= 120 and pred_aop < 120:
                result['FN'] = 1
                result['FN_DATA'] = (label_aop, pred_aop)
            elif label_aop < 120 and pred_aop < 120:
                result['TN'] = 1
            elif label_aop < 120 and pred_aop >= 120:
                result['FP'] = 1
                result['FP_DATA'] = (label_aop, pred_aop)

        result['abs_aop'] = float(aop) if aop is not None else None
        result['pred_aop'] = float(pred_aop_info["aop"]) if pred_aop_info["valid"] else None
        result['label_aop'] = float(label_aop_info["aop"]) if label_aop_info["valid"] else None
        result['pred_aop_valid'] = bool(pred_aop_info["valid"])
        result['label_aop_valid'] = bool(label_aop_info["valid"])
        result['aop_valid'] = bool(aop_valid)
        result['aop_failure_reason'] = self.format_aop_failure_reason(pred_aop_info, label_aop_info)
        if pred_data.shape != label_data.shape:
            raise ValueError(
                f"Prediction/label shape mismatch: {pred_data.shape} vs {label_data.shape}. "
                "Resize predictions to the dataset label resolution before evaluation."
            )

        result.update(self.class_metrics(pred_data == 1, label_data == 1, "ps"))
        result.update(self.class_metrics(pred_data == 2, label_data == 2, "fh"))
        result.update(self.class_metrics(pred_data > 0, label_data > 0, "all"))
        return result

    @staticmethod
    def _numeric_stem(path):
        stem = Path(path).stem
        suffix = stem.split('_')[-1]
        return int(suffix) if suffix.isdigit() else stem

    def resolve_truth_path(self, pred_name):
        pred_path = Path(pred_name)
        exact_path = self.truth_path / pred_path.name
        if exact_path.exists():
            return exact_path

        stem = pred_path.stem.split('_')[-1]
        if stem.isdigit():
            candidates = [self.truth_path / f"{int(stem)}{pred_path.suffix}"]
            candidates.extend(
                self.truth_path / f"{int(stem):0{width}d}{pred_path.suffix}"
                for width in (5, 4, 3)
            )
            for padded_path in candidates:
                if padded_path.exists():
                    return padded_path

        raise FileNotFoundError(
            f"No matching label for prediction {pred_path.name} under {self.truth_path}. "
            "Tried exact name and zero-padded numeric names."
        )

    def process(self):
        names = sorted(
            [name for name in os.listdir(self.pred_path) if not name.startswith('.')],
            key=self._numeric_stem,
        )
        if not names:
            raise ValueError(f"No prediction files found under {self.pred_path}")
        results = []
        for pre_name in tqdm(names):
            pred_image_path = self.pred_path / pre_name
            truth_image_path = self.resolve_truth_path(pre_name)
            pred_image = self.load_image(pred_image_path)
            truth_image = self.load_image(truth_image_path)
            result = self.evaluation(pred_image, truth_image)
            results.append(result)
            name = truth_image_path.stem
            self.results['each_case'][name] = result

        average_score_dict = self.cal_score(results)

        self.results['average'] = average_score_dict
        each_case = self.results['each_case']

        order_dict = {}
        order_dict['average'] = average_score_dict
        order_dict['each_case'] = each_case

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as file:
            json.dump(order_dict, file, indent=2)


    @staticmethod
    def safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    @staticmethod
    def array_to_image(array, spacing=None):
        image = SimpleITK.GetImageFromArray(array.astype(np.uint8))
        if spacing is not None:
            if np.isscalar(spacing):
                spacing = (float(spacing),) * image.GetDimension()
            spacing = tuple(float(value) for value in spacing)
            if len(spacing) != image.GetDimension():
                raise ValueError(
                    f"Expected {image.GetDimension()} spacing values, got {spacing}"
                )
            image.SetSpacing(spacing)
        return image

    def class_metrics(self, pred_mask, label_mask, suffix, spacing=None):
        pred_empty = not bool(pred_mask.any())
        label_empty = not bool(label_mask.any())
        result = {}

        if pred_empty and label_empty:
            result[f'dice_{suffix}'] = 1.0
            result[f'asd_{suffix}'] = 0.0
            result[f'hd_{suffix}'] = 0.0
            return result
        if pred_empty or label_empty:
            result[f'dice_{suffix}'] = 0.0
            result[f'asd_{suffix}'] = 100.0
            result[f'hd_{suffix}'] = 100.0
            return result

        pred_image = self.array_to_image(pred_mask, spacing=spacing)
        label_image = self.array_to_image(label_mask, spacing=spacing)
        result[f'asd_{suffix}'] = float(self.cal_asd(pred_image, label_image))
        result[f'dice_{suffix}'] = float(self.cal_dsc(pred_image, label_image))
        result[f'hd_{suffix}'] = float(self.cal_hd(pred_image, label_image))
        return result

    def cal_asd(self, a, b):
        filter1 = SimpleITK.SignedMaurerDistanceMapImageFilter()  # 于计算二值图像中像素到最近非零像素距离的算法
        filter1.SetUseImageSpacing(True)  # 计算像素距离时要考虑像素之间的间距
        filter1.SetSquaredDistance(False)  # 计算距离时不要对距离进行平方处理
        a_dist = filter1.Execute(a)
        a_dist = SimpleITK.GetArrayFromImage(a_dist)
        a_dist = np.abs(a_dist)
        a_edge = np.zeros(a_dist.shape, a_dist.dtype)
        a_edge[a_dist == 0] = 1
        a_num = np.sum(a_edge)

        filter2 = SimpleITK.SignedMaurerDistanceMapImageFilter()
        filter2.SetUseImageSpacing(True)
        filter2.SetSquaredDistance(False)
        b_dist = filter2.Execute(b)

        b_dist = SimpleITK.GetArrayFromImage(b_dist)
        b_dist = np.abs(b_dist)
        b_edge = np.zeros(b_dist.shape, b_dist.dtype)
        b_edge[b_dist == 0] = 1
        b_num = np.sum(b_edge)

        a_dist[b_edge == 0] = 0.0
        b_dist[a_edge == 0] = 0.0

        denom = a_num + b_num
        if denom == 0:
            return 0.0
        return (np.sum(a_dist) + np.sum(b_dist)) / denom

    def cal_dsc(self, pd, gt):
        pd = SimpleITK.GetArrayFromImage(pd).astype(np.uint8)
        gt = SimpleITK.GetArrayFromImage(gt).astype(np.uint8)
        y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
        return y

    def cal_hd(self, a, b):
        a = SimpleITK.Cast(SimpleITK.RescaleIntensity(a), SimpleITK.sitkUInt8)
        b = SimpleITK.Cast(SimpleITK.RescaleIntensity(b), SimpleITK.sitkUInt8)
        filter1 = SimpleITK.HausdorffDistanceImageFilter()
        filter1.Execute(a, b)
        hd = filter1.GetHausdorffDistance()
        return hd

    @staticmethod
    def format_aop_failure_reason(pred_info, label_info):
        reasons = []
        if not pred_info["valid"]:
            reasons.append(f"pred: {pred_info['reason']}")
        if not label_info["valid"]:
            reasons.append(f"label: {label_info['reason']}")
        return "; ".join(reasons)

    @staticmethod
    def _normalise_ellipse(ellipse):
        (cx, cy), (w, h), angle = ellipse
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        return (float(cx), float(cy)), (w, h), float(angle)

    @staticmethod
    def _largest_component(binary):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return binary
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_label).astype(np.uint8)

    @staticmethod
    def _fit_ellipse_from_points(points, class_name):
        fitters = (cv2.fitEllipseDirect, cv2.fitEllipseAMS, cv2.fitEllipse)
        errors = []
        for fitter in fitters:
            try:
                ellipse = Evaluator._normalise_ellipse(fitter(points))
            except cv2.error as exc:
                errors.append(str(exc).splitlines()[0])
                continue
            if np.isfinite(ellipse[0][0]) and np.isfinite(ellipse[0][1]):
                return ellipse, ""
        return None, f"{class_name} ellipse fitting failed: {' | '.join(errors)}"

    @staticmethod
    def fit_region_ellipse(mask, class_id, class_name, min_area=8):
        binary = (mask == class_id).astype(np.uint8)
        area = int(binary.sum())
        if area == 0:
            return None, f"no {class_name} region"
        if area < min_area:
            return None, f"{class_name} region too small: area={area}"

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if int(binary.sum()) < min_area:
            binary = (mask == class_id).astype(np.uint8)

        component = Evaluator._largest_component(binary)
        component_area = int(component.sum())
        if component_area < min_area:
            return None, f"{class_name} largest component too small: area={component_area}"

        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, f"{class_name} external contour not found"
        contour = max(contours, key=cv2.contourArea)
        if class_name.upper() == "FH" and len(contour) >= 3:
            contour = cv2.convexHull(contour)

        if len(contour) >= 5:
            points = contour.astype(np.float32)
            ellipse, reason = Evaluator._fit_ellipse_from_points(points, class_name)
            if ellipse is not None:
                return ellipse, ""
            return None, reason

        if len(contour) >= 2:
            points = contour.reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(points)
            ellipse = Evaluator._normalise_ellipse(rect)
            return ellipse, f"{class_name} used minAreaRect fallback"
        return None, f"{class_name} contour has too few points for fitting: n={len(contour)}"

    @staticmethod
    def cal_aop_from_mask(mask):
        info = Evaluator.cal_aop_info(mask)
        return info["aop"] if info["valid"] else 0.0

    @staticmethod
    def cal_aop_info(mask):
        if isinstance(mask, SimpleITK.Image):
            mask = SimpleITK.GetArrayFromImage(mask)
        mask = np.asarray(mask).astype(np.uint8)
        ps_ellipse, ps_reason = Evaluator.fit_region_ellipse(mask, 1, "PS")
        fh_ellipse, fh_reason = Evaluator.fit_region_ellipse(mask, 2, "FH")
        reasons = [reason for reason in (ps_reason, fh_reason) if reason and "fallback" not in reason]
        fallback_notes = [reason for reason in (ps_reason, fh_reason) if reason and "fallback" in reason]
        if ps_ellipse is None or fh_ellipse is None:
            reason = "; ".join(reasons or fallback_notes or ["unknown AoP fitting failure"])
            print(f"Warning: AoP ellipse fitting failed ({reason}).")
            return {
                "aop": None,
                "valid": False,
                "reason": reason,
                "ps_ellipse": ps_ellipse,
                "fh_ellipse": fh_ellipse,
                "aop_points": None,
            }
        try:
            geometry = aop_geometry(fh_ellipse, ps_ellipse)
            aop = geometry["aop"]
        except Exception as exc:
            reason = f"AoP geometry failed: {exc}"
            print(f"Warning: {reason}")
            return {
                "aop": None,
                "valid": False,
                "reason": reason,
                "ps_ellipse": ps_ellipse,
                "fh_ellipse": fh_ellipse,
                "aop_points": None,
            }
        if aop is None or not np.isfinite(aop):
            reason = "AoP geometry returned non-finite value"
            print(f"Warning: {reason}")
            return {
                "aop": None,
                "valid": False,
                "reason": reason,
                "ps_ellipse": ps_ellipse,
                "fh_ellipse": fh_ellipse,
                "aop_points": None,
            }
        return {
            "aop": float(aop),
            "valid": True,
            "reason": "; ".join(fallback_notes),
            "ps_ellipse": ps_ellipse,
            "fh_ellipse": fh_ellipse,
            "aop_points": geometry,
        }

    def cal_aop(self, pred):
        return self.cal_aop_from_mask(pred)


    @staticmethod
    def safe_stat(fn, label_aops, pred_aops, name):
        if fn is None:
            print(f"Warning: {name} unavailable because optional dependency import failed: {ICC_IMPORT_ERROR}")
            return None
        if len(label_aops) < 2 or len(pred_aops) < 2:
            print(f"Warning: {name} requires at least two samples. Returning None.")
            return None
        try:
            value = fn(label_aops, pred_aops, verbose=False)
        except Exception as exc:
            print(f"Warning: {name} computation failed: {exc}. Returning None.")
            return None
        if value is None or not np.isfinite(value):
            print(f"Warning: {name} computation returned non-finite value. Returning None.")
            return None
        return float(value)

    def cal_score(self, result):
        m = len(result)
        dice_all_score = 0.
        dice_fh_score = 0.
        dice_ps_score = 0.
        valid_abs_aops = []
        hd_ps_score = 0.
        hd_all_score = 0.
        hd_fh_score = 0.
        asd_all_score = 0.
        asd_fh_score = 0.
        asd_ps_score = 0.
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        fn_list = []
        fp_list = []
        for i in range(m):
            # dice
            dice_all_score += float(result[i].get("dice_all"))
            dice_ps_score += float(result[i].get("dice_ps"))
            dice_fh_score += float(result[i].get("dice_fh"))
            # asa
            asd_all_score += float(result[i].get("asd_all"))
            asd_fh_score += float(result[i].get("asd_fh"))
            asd_ps_score += float(result[i].get("asd_ps"))
            # hd
            hd_all_score += float(result[i].get("hd_all"))
            hd_ps_score += float(result[i].get("hd_ps"))
            hd_fh_score += float(result[i].get("hd_fh"))
            # aop
            if result[i].get("aop_valid") and result[i].get("abs_aop") is not None:
                valid_abs_aops.append(float(result[i].get("abs_aop")))
            tn += int(result[i].get("TN"))
            tp += int(result[i].get("TP"))
            fp += int(result[i].get("FP"))
            fn += int(result[i].get("FN"))
            if result[i].get("FP"):
                fp_list.append(result[i].get("FP_DATA"))
            if result[i].get("FN"):
                fn_list.append(result[i].get("FN_DATA"))
        dice_score = (dice_all_score + dice_ps_score + dice_fh_score) / (3 * m)
        hd_score = (hd_all_score + hd_ps_score + hd_fh_score) / (3 * m)
        asd_score = (asd_all_score + asd_ps_score + asd_fh_score) / (3 * m)

        # score = 0.25 * round(dice_score, 8) + 0.125 * (1 - round(hd_score / 100.0, 8)) + 0.125 * (1 - round(
        # asd_score / 100.0, 8)) + 0.5 * (1 - aop_score / 180.0)

        aggregates = dict()
        outlier_threshold = 30.0
        valid_abs_aops_np = np.array(valid_abs_aops, dtype=np.float32)
        valid_aop_items = [item for item in result if item.get("aop_valid")]
        pred_aops = [float(item.get("pred_aop")) for item in valid_aop_items]
        label_aops = [float(item.get("label_aop")) for item in valid_aop_items]
        aggregates['aop_valid_count'] = len(valid_aop_items)
        aggregates['aop_invalid_count'] = m - len(valid_aop_items)
        aggregates['pred_aop_invalid_count'] = sum(not item.get("pred_aop_valid") for item in result)
        aggregates['label_aop_invalid_count'] = sum(not item.get("label_aop_valid") for item in result)
        aggregates['abs_aop'] = float(valid_abs_aops_np.mean()) if len(valid_abs_aops_np) else None
        aggregates['median_abs_aop'] = float(np.median(valid_abs_aops_np)) if len(valid_abs_aops_np) else None
        aggregates['p90_abs_aop'] = float(np.percentile(valid_abs_aops_np, 90)) if len(valid_abs_aops_np) else None
        aggregates['aop_outlier_count'] = int(np.sum(valid_abs_aops_np >= outlier_threshold)) if len(valid_abs_aops_np) else 0
        aggregates['aop_outlier_threshold'] = outlier_threshold
        aggregates['r_aop'] = self.safe_stat(pearson, label_aops, pred_aops, 'Pearson')
        aggregates['icc_aop'] = self.safe_stat(icc, label_aops, pred_aops, 'ICC')
        aggregates['dice_ps'] = dice_ps_score / m
        aggregates['dice_fh'] = dice_fh_score / m
        aggregates['dice_all'] = dice_all_score / m
        aggregates['hd_ps'] = hd_ps_score / m
        aggregates['hd_fh'] = hd_fh_score / m
        aggregates['hd_all'] = hd_all_score / m
        aggregates['asd_ps'] = asd_ps_score / m
        aggregates['asd_fh'] = asd_fh_score / m
        aggregates['asd_all'] = asd_all_score / m
        aggregates['TN'] = tn
        aggregates['TP'] = tp
        aggregates['FN'] = fn
        aggregates['FP'] = fp
        aggregates['precision'] = self.safe_divide(tp, tp + fp)
        aggregates['recall'] = self.safe_divide(tp, tp + fn)
        aggregates['f1-score'] = self.safe_divide(
            2 * aggregates['precision'] * aggregates['recall'],
            aggregates['precision'] + aggregates['recall'],
        )
        aggregates['specificity'] = self.safe_divide(tn, fp + tn)
        aggregates['wuzhen'] = 1 - aggregates['specificity']
        aggregates['louzhen'] = 1 - aggregates['recall']
        aggregates['fp_list'] = fp_list
        aggregates['fn_list'] = fn_list
        return aggregates


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BANet prediction masks.")
    parser.add_argument("--preds-dir", required=True, help="Directory containing predicted class-index masks.")
    parser.add_argument("--truth-dir", required=True, help="Directory containing ground-truth class-index masks.")
    parser.add_argument("--save-path", required=True, help="Path to save the JSON evaluation report.")
    return parser.parse_args()


# python -m eval.evaluation --preds-dir pred/iugc --truth-dir ./data/iugc/label --save-path ./iugc.json

if __name__ == "__main__":
    args = parse_args()
    Evaluator(args.preds_dir, args.truth_dir, args.save_path).process()


# if __name__ == "__main__":
#     preds_dir = f"./pred/psfhs1"
#     truth_dir = f"./data/test/testdataset/label"
#     save_path = f"./psfhs1.json"
#     Evaluator(preds_dir, truth_dir, save_path).process()


# if __name__ == "__main__":
#     preds_dir = f"./pred/psfhs2"
#     truth_dir = f"./data/psfhs2/label"
#     save_path = f"./psfhs2.json"
#     Evaluator(preds_dir, truth_dir, save_path).process()
