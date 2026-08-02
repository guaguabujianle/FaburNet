import pandas as pd
import pingouin as pg
import numpy as np
import json
import os

def get_data(model):
    label = []
    pred = []

    file = open(f"./details/psfhs/{model}.json")
    data = json.load(file)['each_case']
    """
    for i in range(1, 701):
        label.append(data[str(i)]["label_aop"])
        pred.append(data[str(i)]["pred_aop"])
    

    file = open(f"./details/iugc/{model}.json")
    data = json.load(file)['each_case']
    for i in range(1, 301):
        label.append(data[str(i)]["label_aop"])
        pred.append(data[str(i)]["pred_aop"])
    """
    file = open(f"./details/psfhs_val/{model}.json")
    data = json.load(file)['each_case']
    for i in range(1, 402):
        label.append(data[str(i)]["label_aop"])
        pred.append(data[str(i)]["pred_aop"])

    return label, pred

def icc(label, pred, verbose=True):
    label = [float(v) for v in label]
    pred = [float(v) for v in pred]
    samples = range(1, len(label) + 1)

    # 转换为 long format
    df = pd.DataFrame({
        'sample': list(samples) * 2,
        'rater': ['A'] * len(samples) + ['B'] * len(samples),
        'score': label + pred
    })

    icc_table = pg.intraclass_corr(data=df, targets='sample', raters='rater', ratings='score')
    icc3 = icc_table[icc_table['Type'].isin(['ICC3', 'ICC(C,1)'])]
    if icc3.empty:
        raise ValueError(f"ICC3 / ICC(C,1) row not found. Available types: {icc_table['Type'].tolist()}")
    if verbose:
        print(icc3)
    return float(icc3['ICC'].iloc[0])

def pearson(label, pred, verbose=True):
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(label, pred)
    if verbose:
        print("Pearson correlation coefficient:", corr)
        print("P-value:", p_value)
    return float(corr)

if __name__ == '__main__':
    models = [i.split(".")[0] for i in os.listdir("./details/psfhs")]
    for model in models:
        label,pred = get_data(model)
        print(model)
        icc(label,pred)
        pearson(label,pred)



