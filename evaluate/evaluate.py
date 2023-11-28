from torchmetrics import F1Score, Accuracy, Precision, Recall

def compute_score(num_classes, y_true, y_pred):
    Acc_fn = Accuracy(task= "multiclass", num_classes= num_classes)
    Prec_fn = Precision(task= "multiclass", num_classes= num_classes)
    Recall_fn = Recall(task= "multiclass", num_classes= num_classes)
    F1_score = F1Score(task= "multiclass", num_classes= num_classes, average= 'macro')

    acc = Acc_fn(y_true, y_pred)
    prec = Prec_fn(y_true, y_pred)
    recall = Recall_fn(y_true, y_pred)
    f1 = F1_score(y_true, y_pred)

    return acc, prec, recall, f1