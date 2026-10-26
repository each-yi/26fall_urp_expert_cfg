"""Explicit evaluation conventions; missing paper details are documented, not guessed silently."""
import re
from collections import Counter
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve


def tokens(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def score(answer, prediction, answer_type):
    gt, pred = tokens(answer), tokens(prediction)
    if answer_type == 'closed':
        # First explicit yes/no; avoids crediting both answers in a long response.
        labels = [t for t in pred if t in ('yes','no')]
        return float(bool(labels) and labels[0] == ' '.join(gt))
    counts = Counter(gt)
    return sum((counts & Counter(pred)).values()) / len(gt) if gt else 0.


def ece(y, confidence, bins=15):
    y, p = np.asarray(y), np.asarray(confidence)
    idx = np.minimum((p * bins).astype(int), bins-1)
    return float(sum(np.mean(idx == i) * abs(y[idx == i].mean()-p[idx == i].mean())
                     for i in range(bins) if np.any(idx == i)))


def fit_temperature(y, confidence):
    if len(y) == 0 or len(set(y)) < 2:
        raise ValueError('Calibration validation set must contain correct and incorrect answers')
    z = logit(np.clip(confidence,1e-7,1-1e-7))
    def loss(log_t):
        x = z / np.exp(log_t)
        return np.mean(np.logaddexp(0,x)-np.asarray(y)*x)
    return float(np.exp(minimize_scalar(loss,bounds=(-5,5),method='bounded').x))


def calibration(y, confidence, temperature, bins=15):
    p = np.asarray(confidence, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(y) == 0 or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid/empty confidence input')
    pt = expit(logit(np.clip(p,1e-7,1-1e-7))/temperature)
    auc = float(roc_auc_score(y,p)) if len(set(y)) == 2 else None
    return {'ece':ece(y,p,bins), 'ece_t':ece(y,pt,bins),
            'bs_t':float(np.mean((pt-y)**2)), 'auc':auc}


def uncertainty(y_error, entropy):
    if len(set(y_error)) < 2:
        return {'auc':None,'roc':[], 'reason':'Only one correctness class'}
    fpr,tpr,thresholds = roc_curve(y_error,entropy)
    return {'auc':float(roc_auc_score(y_error,entropy)),
            'roc':[{'threshold':float(t) if np.isfinite(t) else None,
                    'sensitivity':float(s),'specificity':float(1-f)}
                   for f,s,t in zip(fpr,tpr,thresholds)]}
