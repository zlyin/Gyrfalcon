## import packages
import torch
from torch import nn
import torch.nn.functional as F


"""
- define a class of NMS class
- inputs:
    - bboxes = proposed bounding boxes for suppression
    - scores = cls scores to distinguish fg, bg object
    - thres = IoU thres value, default=0.5
"""

def NMS(self, bboxes, scores, thres=0.5):
    # get (x1, y1) & (x2, y2) of bboxes 
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # calcualte boxes area
    area = (x2 - x1 + 1) * (y2 - y1 + 1) 

    # sort cls scores in descending order
    _, order = scores.sort(0, descending=True)
    # placeholder for keep bboxes
    keep = []

    # non-maximum suppression iteratively
    while order.numel() > 0:
        if order.numel() == 1:  # only 1 box left
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item() # set max score box as the current box
            keep.append(i)
        
        # tensor.clamp bboxes to coords of current box
        xx1 = x1[order[1:]].clamp(min=x1[i])
        xx2 = x2[order[1:]].clamp(min=x2[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        yy2 = y2[order[1:]].clamp(min=y2[i])
        # get interception area between bboxes and the current box
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        # calculate IoUs
        iou = inter / (area[i] + area[order[1:]] - inter)
        
        # keep bboxes with IoU < thres value
        index = (iou < thres).nonzero().squeeze()   # kill if IoU > thres
        if index.numel() == 0:
            break
        
        # refresh the order, narrow down to a smaller size for next interation
        order = order[index + 1]    # +1 to remove the gap btw index & order
        pass
    # return filtered bboxes
    return torch.LongTensor(keep)


