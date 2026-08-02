import json
import numpy as np

PER_CASE_METRICS = [
    "abs_aop",
    "asd_ps", "dice_ps", "hd_ps",
    "asd_fh", "dice_fh", "hd_fh",
    "asd_all", "dice_all", "hd_all",
]
AGGREGATE_METRICS = ["r_aop", "icc_aop"]


def load_result(fix):
    with open(f"./{fix}.json", "r") as file:
        return json.load(file)


def get_all_fold(fix, metric="abs_aop"):
    ls = []
    for i in range(5):
        with open(f"../output/ablation/{fix}/{fix}_fold{i}.json", "r") as file:
            data = json.load(file)
        if metric in AGGREGATE_METRICS:
            ls.append(float(data["average"][metric]))
        else:
            dic = data["each_case"]
            ls.append([float(value[metric]) for case, value in dic.items()])
    return ls


def get_one_fold(fix, metric="abs_aop"):
    data = load_result(fix)
    if metric in AGGREGATE_METRICS:
        return float(data["average"][metric])
    dic = data["each_case"]
    return [float(value[metric]) for case, value in dic.items()]


def cal_mean_std_one(fix):
    for metric in PER_CASE_METRICS:
        ls = get_one_fold(fix, metric)
        print(f"{metric}: mean={np.mean(ls):.4f}, std={np.std(ls):.4f}\n")

    for metric in AGGREGATE_METRICS:
        value = get_one_fold(fix, metric)
        print(f"{metric}: value={value:.4f}\n")


if __name__ == "__main__":
    fix = "psfhs1"
    cal_mean_std_one(fix)
