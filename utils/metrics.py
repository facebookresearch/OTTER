from utils.utils import AverageMeter


class Accuracy:
    def __init__(self):
        self.top1 = AverageMeter()

    def update(self, targets, preds):
        prec1 = sum(preds == targets) / len(targets)
        self.top1.update(prec1, len(targets))

    def avg(self):
        return self.top1.avg

    @property
    def prec1(self):
        return self.top1.avg

    def __str__(self):
        return "Accuracy"


class FAtK:
    def __init__(self, K):
        """
        K is a comma-separated string of ints
        e.g. 1,2,5,10
        """
        K = K.split(",")
        self.K2Meter = {int(k): AverageMeter() for k in K}

    def update(self, targets, preds, k):
        _, pred_matrix = preds
        batch_size = pred_matrix.shape[0]

        count = 0
        for i in range(batch_size):
            row_pred = pred_matrix[i]
            row_label = targets[i]
            if len(set(row_pred).intersection(row_label)) != 0:
                count += 1
        self.K2Meter[k].update(count / batch_size, batch_size)

    def avg(self):
        return " | ".join([f"top{k}: {meter.avg:.5f}" for k, meter in self.K2Meter.items()])

    @property
    def prec1(self):
        if 1 not in self.K2Meter:
            return 0
        return self.K2Meter[1].avg

    @property
    def Ks(self):
        return self.K2Meter.keys()

    def __str__(self):
        return "FAtK"
