import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluation import Evaluator
from models.faburnet import FaburNet
from utils.erase_dataloader import JointTransform2D, correct_dims

METRIC_COLUMNS = [
    "dice_ps",
    "dice_fh",
    "dice_all",
    "asd_ps",
    "asd_fh",
    "asd_all",
    "hd_ps",
    "hd_fh",
    "hd_all",
    "abs_aop",
    "pred_aop",
    "label_aop",
    "pred_aop_valid",
    "label_aop_valid",
    "aop_valid",
    "aop_failure_reason",
]

SUMMARY_NUMERIC_COLUMNS = [
    "dice_ps",
    "dice_fh",
    "dice_all",
    "asd_ps",
    "asd_fh",
    "asd_all",
    "hd_ps",
    "hd_fh",
    "hd_all",
    "abs_aop",
    "pred_aop",
    "label_aop",
]

SUMMARY_RATE_COLUMNS = ["pred_aop_valid", "label_aop_valid", "aop_valid"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_files: np.ndarray
    label_files: np.ndarray
    eval_size: tuple[int, int] | None = None
    image_channel_first: bool = False
    output_name_from_int_stem: bool = False


class EvalDataset(Dataset):
    def __init__(self, spec: DatasetSpec, image_size=256):
        self.spec = spec
        self.image_size = image_size
        self.transform = JointTransform2D(
            img_size=image_size,
            low_img_size=128,
            ori_size=image_size,
            crop=None,
            p_flip=0.0,
            color_jitter_params=None,
            long_mask=True,
        )
        self.images = [self.read_array(path, resize_to=(image_size, image_size)) for path in spec.image_files]
        self.labels_for_transform = [
            self.read_array(path, resize_to=(image_size, image_size)).astype(np.uint8)
            for path in spec.label_files
        ]
        self.output_names = [self.output_name(path) for path in spec.label_files]

    @staticmethod
    def read_array(path, resize_to=None):
        image = sitk.ReadImage(str(path))
        if resize_to is not None and tuple(image.GetSize())[:2] != tuple(resize_to):
            image = resize_image_itk(image, resize_to)
        return sitk.GetArrayFromImage(image)

    def output_name(self, path):
        stem = Path(path).stem
        if self.spec.output_name_from_int_stem and stem.isdigit():
            return str(int(stem))
        return stem

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.spec.image_channel_first and image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose((1, 2, 0))
        image, mask = correct_dims(image, self.labels_for_transform[idx])
        image, mask, _ = self.transform(image, mask)
        return {
            "image": image,
            "case_id": self.output_names[idx],
            "image_path": str(self.spec.image_files[idx]),
            "label_path": str(self.spec.label_files[idx]),
        }


def resize_image_itk(itkimage, new_size, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    origin_size = np.array(itkimage.GetSize(), dtype=np.float64)
    origin_spacing = np.array(itkimage.GetSpacing(), dtype=np.float64)
    new_size = np.array(new_size, dtype="uint32")
    new_spacing = origin_spacing * (origin_size / new_size)

    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(new_size.tolist())
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetTransform(
        sitk.Transform(itkimage.GetDimension(), sitk.sitkIdentity)
    )
    resampler.SetInterpolator(resamplemethod)
    resized = resampler.Execute(itkimage)
    resized.SetOrigin(itkimage.GetOrigin())
    resized.SetSpacing(itkimage.GetSpacing())
    resized.SetDirection(itkimage.GetDirection())
    return resized


def resize_mask_array(mask, size):
    if size is None or tuple(mask.shape[-2:]) == tuple(size[::-1]):
        return mask.astype(np.uint8)
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    image = resize_image_itk(image, size)
    return sitk.GetArrayFromImage(image).astype(np.uint8)


def read_test_indices(test_file):
    test_file = Path(test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test split file not found: {test_file}")
    with open(test_file, "r") as file:
        lines = file.readlines()
    return np.array([int(line.strip().split("/")[-1]) - 1 for line in lines], dtype=np.int64)


def require_files(paths, dataset_name, kind):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        preview = "\n".join(missing[:5])
        extra = "" if len(missing) <= 5 else f"\n... and {len(missing) - 5} more"
        raise FileNotFoundError(f"Missing {kind} files for {dataset_name}:\n{preview}{extra}")


def make_dataset_specs(root: Path, names=None):
    names = list(names or ("test", "psfhs", "iugc"))
    specs = {}

    data_root = root / "dataset" / "data"

    if "test" in names:
        test_root = data_root / "test" / "testdataset"
        test_indices = read_test_indices(data_root / "test" / "test.txt")
        test_images = np.array([
            test_root / "image_mha" / f"{idx:05d}.mha" for idx in range(1, 701)
        ])
        test_labels = np.array([
            test_root / "label_mha" / f"{idx:05d}.mha" for idx in range(1, 701)
        ])
        specs["test"] = DatasetSpec(
            name="test",
            image_files=test_images[test_indices],
            label_files=test_labels[test_indices],
            eval_size=(256, 256),
            image_channel_first=True,
        )

    if "psfhs" in names:
        psfhs_root = data_root / "psfhs"
        psfhs_indices = read_test_indices(psfhs_root / "test.txt")
        psfhs_images = np.array([
            psfhs_root / "img" / f"{idx:05d}.png" for idx in range(1, 402)
        ])
        psfhs_labels = np.array([
            psfhs_root / "label" / f"{idx}.png" for idx in range(1, 402)
        ])
        specs["psfhs"] = DatasetSpec(
            name="psfhs",
            image_files=psfhs_images[psfhs_indices],
            label_files=psfhs_labels[psfhs_indices],
            eval_size=(256, 256),
            output_name_from_int_stem=True,
        )

    if "iugc" in names:
        iugc_root = data_root / "iugc"
        iugc_indices = read_test_indices(iugc_root / "test.txt")
        iugc_images = np.array([
            iugc_root / "img" / f"{idx:05d}.png" for idx in range(1, 301)
        ])
        iugc_labels = np.array([
            iugc_root / "label" / f"{idx:05d}.png" for idx in range(1, 301)
        ])
        specs["iugc"] = DatasetSpec(
            name="iugc",
            image_files=iugc_images[iugc_indices],
            label_files=iugc_labels[iugc_indices],
            eval_size=(512, 512),
        )

    available = {"test", "psfhs", "iugc"}
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}. Available: {sorted(available)}")

    for spec in specs.values():
        require_files(spec.image_files, spec.name, "image")
        require_files(spec.label_files, spec.name, "label")
    return specs


def load_model(checkpoint_path, device, weights_name="ema_model"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and weights_name in checkpoint:
        state_dict = checkpoint[weights_name]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        print(
            f"Checkpoint has no '{weights_name}' entry; falling back to 'model'."
        )
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model state dictionary")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model = FaburNet()
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def final_logits(model_output):
    """Extract FaburNet's final segmentation logits from deep-supervision output."""
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[-1]
    if not isinstance(model_output, torch.Tensor) or model_output.ndim != 4:
        raise TypeError("FaburNet output does not contain [B, C, H, W] logits")
    return model_output


def evaluate_prediction(evaluator, pred_mask, label_path):
    label = sitk.ReadImage(str(label_path))
    pred = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    return evaluator.evaluation(pred, label)


def image_to_rgb(image_path, size):
    image = sitk.ReadImage(str(image_path))
    if size is not None and tuple(image.GetSize())[:2] != tuple(size):
        image = resize_image_itk(image, size, resamplemethod=sitk.sitkLinear)
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = array.transpose((1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = array.astype(np.float32)
    out = np.zeros_like(array, dtype=np.uint8)
    for c in range(3):
        channel = array[:, :, c]
        lo, hi = np.percentile(channel, [1, 99])
        if hi <= lo:
            hi = channel.max()
            lo = channel.min()
        if hi > lo:
            channel = (channel - lo) / (hi - lo)
        else:
            channel = channel * 0
        out[:, :, c] = np.clip(channel * 255, 0, 255).astype(np.uint8)
    return out


def as_int_point(point):
    return int(round(point[0])), int(round(point[1]))


def draw_point(canvas, point, color):
    p = as_int_point(point)
    cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, p, 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_aop_geometry(canvas, info):
    geometry = info.get("aop_points")
    if not geometry:
        return
    ps_left = geometry["ps_left"]
    ps_right = geometry["ps_right"]
    fh_contact = geometry["fh_contact"]
    line_color = (0, 255, 0)
    point_color = (255, 0, 255)

    cv2.line(canvas, as_int_point(ps_left), as_int_point(ps_right), line_color, 2, cv2.LINE_AA)
    cv2.line(canvas, as_int_point(ps_right), as_int_point(fh_contact), line_color, 2, cv2.LINE_AA)
    draw_point(canvas, ps_left, point_color)
    draw_point(canvas, ps_right, point_color)
    draw_point(canvas, fh_contact, point_color)

    vertex = as_int_point(ps_right)
    aop_text = f"AoP={geometry['aop']:.2f}"
    cv2.putText(canvas, aop_text, (vertex[0] + 10, vertex[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, aop_text, (vertex[0] + 10, vertex[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 1, cv2.LINE_AA)


def draw_ellipse_info(canvas, info):
    ps_color = (0, 255, 255)
    fh_color = (0, 0, 255)
    if info.get("ps_ellipse") is not None:
        cv2.ellipse(canvas, info["ps_ellipse"], ps_color, 2)
    if info.get("fh_ellipse") is not None:
        cv2.ellipse(canvas, info["fh_ellipse"], fh_color, 2)
    draw_aop_geometry(canvas, info)


def overlay_pred_mask(canvas, pred_mask, alpha=0.35):
    color_mask = np.zeros_like(canvas, dtype=np.uint8)
    color_mask[pred_mask == 1] = (0, 255, 255)
    color_mask[pred_mask == 2] = (255, 0, 0)
    foreground = pred_mask > 0
    blended = cv2.addWeighted(canvas, 1.0 - alpha, color_mask, alpha, 0)
    canvas[foreground] = blended[foreground]


def save_aop_overlay(image_path, pred_mask, label_path, output_path, eval_size):
    label_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.uint8)
    if pred_mask.shape != label_mask.shape:
        pred_mask = resize_mask_array(pred_mask, eval_size)
    base = image_to_rgb(image_path, eval_size or tuple(label_mask.shape[::-1]))
    if base.shape[:2] != label_mask.shape:
        base = cv2.resize(base, tuple(label_mask.shape[::-1]), interpolation=cv2.INTER_LINEAR)
    overlay = base.copy()
    overlay_pred_mask(overlay, pred_mask)
    pred_info = Evaluator.cal_aop_info(pred_mask)
    draw_ellipse_info(overlay, pred_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def parse_float(value):
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def summarize_result_csv(csv_path):
    with open(csv_path, "r", newline="") as file:
        rows = list(csv.DictReader(file))

    summary = {"dataset": csv_path.stem.split("_")[-1], "cases": len(rows)}
    for column in SUMMARY_NUMERIC_COLUMNS:
        values = [parse_float(row.get(column)) for row in rows]
        values = [value for value in values if value is not None]
        summary[column] = float(np.mean(values)) if values else None
        summary[f"_{column}_count"] = len(values)

    for column in SUMMARY_RATE_COLUMNS:
        values = [parse_bool(row.get(column)) for row in rows]
        values = [value for value in values if value is not None]
        rate_column = f"{column}_rate"
        summary[rate_column] = float(np.mean(values)) if values else None
        summary[f"_{rate_column}_count"] = len(values)

    return summary


def format_summary_value(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def print_evaluation_summary(csv_paths):
    summaries = [summarize_result_csv(path) for path in csv_paths if path.exists()]
    if not summaries:
        print("No result CSV files found for summary.")
        return

    overall = {"dataset": "overall", "cases": sum(item["cases"] for item in summaries)}
    for column in SUMMARY_NUMERIC_COLUMNS:
        weighted_sum = sum(
            item[column] * item[f"_{column}_count"]
            for item in summaries
            if item[column] is not None
        )
        count = sum(item[f"_{column}_count"] for item in summaries)
        overall[column] = weighted_sum / count if count else None
    for column in SUMMARY_RATE_COLUMNS:
        rate_column = f"{column}_rate"
        weighted_sum = sum(
            item[rate_column] * item[f"_{rate_column}_count"]
            for item in summaries
            if item[rate_column] is not None
        )
        count = sum(item[f"_{rate_column}_count"] for item in summaries)
        overall[rate_column] = weighted_sum / count if count else None

    print("\nEvaluation summary (average per case):")
    header = [
        "dataset",
        "cases",
        "dice_ps",
        "dice_fh",
        "dice_all",
        "asd_all",
        "hd_all",
        "abs_aop",
        "aop_valid_rate",
    ]
    print(", ".join(header))
    for item in [*summaries, overall]:
        print(
            ", ".join(
                [
                    str(item["dataset"]),
                    str(item["cases"]),
                    format_summary_value(item["dice_ps"]),
                    format_summary_value(item["dice_fh"]),
                    format_summary_value(item["dice_all"]),
                    format_summary_value(item["asd_all"]),
                    format_summary_value(item["hd_all"]),
                    format_summary_value(item["abs_aop"]),
                    format_summary_value(item["aop_valid_rate"]),
                ]
            )
        )


def evaluate_checkpoint(
    checkpoint_path,
    dataset_loaders,
    dataset_specs,
    result_dir,
    visualization_dir,
    device,
    weights_name="ema_model",
    save_visualizations=False,
):
    print(f"Loading checkpoint: {checkpoint_path.name}")
    model = load_model(checkpoint_path, device, weights_name=weights_name)
    checkpoint_stem = checkpoint_path.stem

    for dataset_name, loader in dataset_loaders.items():
        spec = dataset_specs[dataset_name]
        csv_path = result_dir / f"FaburNet_{checkpoint_stem}_{dataset_name}.csv"
        dataset_visualization_dir = visualization_dir / dataset_name / checkpoint_stem
        evaluator = Evaluator(ROOT, Path(spec.label_files[0]).parent, csv_path)

        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["case_id", "checkpoint", "dataset", *METRIC_COLUMNS])
            writer.writeheader()

            iterator = tqdm(loader, desc=f"{checkpoint_path.name} [{dataset_name}]", leave=False)
            with torch.no_grad():
                for batch in iterator:
                    images = batch["image"].to(device=device, dtype=torch.float32)
                    logits = final_logits(model(images))
                    labels = batch["label_path"]
                    images_path = batch["image_path"]
                    case_ids = batch["case_id"]

                    for i in range(logits.shape[0]):
                        pred_mask = logits[i].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
                        pred_mask = resize_mask_array(pred_mask, spec.eval_size)
                        metrics = evaluate_prediction(evaluator, pred_mask, labels[i])
                        if save_visualizations:
                            save_aop_overlay(
                                images_path[i],
                                pred_mask,
                                labels[i],
                                dataset_visualization_dir / f"{case_ids[i]}.png",
                                spec.eval_size,
                            )
                        row = {
                            "case_id": str(case_ids[i]),
                            "checkpoint": checkpoint_path.name,
                            "dataset": dataset_name,
                        }
                        row.update({key: metrics[key] for key in METRIC_COLUMNS})
                        writer.writerow(row)
        print(f"Saved {csv_path}")

    print_evaluation_summary([
        result_dir / f"FaburNet_{checkpoint_stem}_{dataset_name}.csv"
        for dataset_name in dataset_loaders
    ])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a FaburNet checkpoint on the test, PSFHS, and IUGC datasets."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "checkpoints/faburnet_vNone"
            / "epoch50_0.04182_0.93866_0.85838_0.96581_0.93025.pth"
        ),
        help="FaburNet checkpoint path.",
    )
    parser.add_argument("--result-dir", default=str(ROOT / "eval_results"))
    parser.add_argument("--visualization-dir", default=str(ROOT / "eval_visualizations"))
    parser.add_argument("--save-visualizations", action="store_true", help="Save AoP overlay images under --visualization-dir.")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu. Defaults to CUDA when available.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--weights",
        choices=("ema_model", "model"),
        default="ema_model",
        help="Checkpoint state dictionary to evaluate.",
    )
    parser.add_argument("--datasets", nargs="+", default=["test", "psfhs", "iugc"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-incompatible", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    visualization_dir = Path(args.visualization_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if checkpoint_path.is_dir():
        raise IsADirectoryError(f"--checkpoint must be a file, got directory: {checkpoint_path}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    selected_specs = make_dataset_specs(ROOT, args.datasets)
    dataset_loaders = {
        name: DataLoader(
            EvalDataset(spec),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, spec in selected_specs.items()
    }

    expected_csvs = [
        result_dir / f"FaburNet_{checkpoint_path.stem}_{name}.csv"
        for name in selected_specs
    ]
    if args.skip_existing and all(path.exists() for path in expected_csvs):
        print(f"Skipping existing results for {checkpoint_path.name}")
        print_evaluation_summary(expected_csvs)
        return

    try:
        evaluate_checkpoint(
            checkpoint_path,
            dataset_loaders,
            selected_specs,
            result_dir,
            visualization_dir,
            device,
            weights_name=args.weights,
            save_visualizations=args.save_visualizations,
        )
    except RuntimeError as exc:
        if args.skip_incompatible:
            print(f"Skipping incompatible checkpoint {checkpoint_path.name}: {exc}")
            return
        raise


if __name__ == "__main__":
    main()
