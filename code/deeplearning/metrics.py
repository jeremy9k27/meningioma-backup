import numpy as np

def basic_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).int()
    correct = (y_pred == y_true.int()).sum()
    total = y_true.numel()
    return correct.float() / total

def balanced_accuracy(y_true, y_pred):
    y_pred = (y_pred > 0.5).int()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))

def true_positive_rate(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).int()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn)

def false_positive_rate(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).int()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return fp / (fp + tn)

def false_discovery_rate(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).int()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return fp / (fp + tp)

def tpr_at_fpr(y_true, y_pred, target_fpr=0.1):
    thresholds = np.linspace(0, 1, 100)
    fprs = []

    for t in thresholds:
        fpr = false_positive_rate(y_true, y_pred, threshold=t)
        fprs.append(fpr)

    # Replace NaNs with inf to prevent invalid argmin results
    fprs = np.where(np.isnan(fprs), np.inf, fprs)

    # Compute the closest index and ensure it's in valid bounds
    th_idx = np.argmin(np.abs(fprs - target_fpr))
    th_idx = np.clip(th_idx, 0, len(thresholds) - 1)

    return true_positive_rate(y_true, y_pred, threshold=thresholds[th_idx])