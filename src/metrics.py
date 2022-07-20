import torch
import torch.nn as nn
import math


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = ((outputs == labels) * labels).float()
        # print(torch.sum(true_positive))
        # print(torch.sum(outputs == labels))
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)
        # print(recall,precision)
        return precision, recall
        
class CalcBeta(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(CalcBeta, self).__init__()
        self.threshold = threshold

    def __call__(self, targets):
        edge_targets = (targets > self.threshold)

        total_et = torch.sum(edge_targets.float())
        whole = targets.shape[2]*targets.shape[3]

        beta = (whole - total_et) / whole

        return beta

class DepthAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(DepthAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):

        loss = nn.L1Loss()
        loss_2 = nn.MSELoss()
        mae = loss(inputs, outputs)
        mse = loss_2(inputs, outputs)
        return mse , mae

class SegAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """
    def __init__(self, threshold=0.5):
        super(SegAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):

        loss = nn.L1Loss()
        loss_2 = nn.MSELoss()
        mae = loss(inputs, outputs)
        mse = loss_2(inputs, outputs)
        return mse , mae

class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()
        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return torch.tensor(0)

        return self.max_val - 10 * torch.log(mse) / self.base10
        