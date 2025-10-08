import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanBoxTracker:
    """
    Simplified Kalman tracker for a bounding box.
    """
    count = 0
    def __init__(self, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1,y1,x2,y2]
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1
        self.no_losses = 0

    def update(self, bbox):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.hits += 1
        self.no_losses = 0

    def predict(self):
        # In this simplified version, no motion prediction
        self.no_losses += 1
        return self.bbox

class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    @staticmethod
    def iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
                 + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
        return o

    def update(self, dets=np.empty((0,5))):
        """
        Params:
          dets - np.array of detections in format [[x1,y1,x2,y2,score],...]
        Returns:
          np.array of tracked objects [[x1,y1,x2,y2,id],...]
        """
        trks = np.array([trk.predict() for trk in self.trackers])
        matched, unmatched_dets, unmatched_trks = [], [], []

        if len(trks)>0 and len(dets)>0:
            iou_matrix = np.zeros((len(trks), len(dets)), dtype=np.float32)
            for t,trk in enumerate(trks):
                for d,det in enumerate(dets):
                    iou_matrix[t,d] = self.iou(trk, det[:4])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r,c in zip(row_ind,col_ind):
                if iou_matrix[r,c]<self.iou_threshold:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    matched.append((r,c))
            unmatched_trks += [i for i in range(len(trks)) if i not in row_ind]
            unmatched_dets += [i for i in range(len(dets)) if i not in col_ind]
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(trks)))

        # Update matched trackers
        for t,d in matched:
            self.trackers[t].update(dets[d][:4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i][:4]))

        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.no_losses > self.max_age:
                self.trackers.pop(i-1)
            i -= 1

        # Output
        ret = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits:
                ret.append(list(trk.bbox)+[trk.id])
        return np.array(ret)
