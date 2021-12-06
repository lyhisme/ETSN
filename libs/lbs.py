import torch
import numpy as np
from libs.metric import get_segments
from utils.class_id_map import get_id2class_map

def LBS(pred, output, config):

    output = output.data.squeeze(0).cpu().numpy()
    id2class_map = get_id2class_map(config.dataset, dataset_dir = config.dataset_dir)
    # remove short burrs
    p_label, p_start, p_end = get_segments(pred, id2class_map)
    for i in range(len(p_label)):
        if ((p_end[i] - p_start[i]) <= config.lbs_burr):
            if(i == 0):
                for k in range(p_end[i], p_start[i]-1, -1):
                    pred[k] = pred[k+1]               
            else: 
                for k in range(p_start[i], p_end[i]+1):
                    pred[k] = pred[k-1]

    confidence_burr = []
    p_label, p_start, p_end = get_segments(pred, id2class_map)
    # locate the burrs
    for i in range(len(p_label)):
        if ((p_end[i] - p_start[i]) <= config.lbs_window ): 
            sum =[]
            for m in range(p_start[i],p_end[i]):
                sum.append(np.max(output[:,m:m+1]))
            if(np.mean(sum) <= config.lbs_Confidence):
                confidence_burr.append(1)
            else:
                confidence_burr.append(0)
        else:
            confidence_burr.append(0)

    count, old_index, consecutive_index= 0, 0, 0
    while (count < len(confidence_burr)):
        if (confidence_burr[count]):
            consecutive_index = 0
            old_index = count
            while confidence_burr[count]:
                consecutive_index += 1
                count += 1
                if (count >= len(confidence_burr)):
                    break
            for j in range (consecutive_index):
                confidence_burr[old_index+j] = consecutive_index
        else:
            count += 1

    index = 0
    # remove consecutive burrs
    while (index < len(confidence_burr)):
        if(confidence_burr[index] > 1):
            if(index + confidence_burr[index]-1 == len(confidence_burr)-1):
                for k in range(p_start[index], p_end[index + int(confidence_burr[index])-1]+1):
                    pred[k] = pred[p_start[index]-1]
            elif(index + confidence_burr[index]-1 == 0):
                for k in range(p_end[index + int(confidence_burr[index])-1], p_start[index]-1, -1):
                    pred[k] = pred[p_end[index + int(confidence_burr[index])-1]+1]
            else:
                if (confidence_burr[index] % 2 == 0):
                    for k in range(p_start[index], p_end[index + int(confidence_burr[index]/2)-1]+1):
                        pred[k] = pred[p_start[index]-1]
                    for k in range(p_end[index + int(confidence_burr[index])-1], p_end[index + int(confidence_burr[index]/2)-1]-1, -1):
                        pred[k] = pred[p_end[index + int(confidence_burr[index])-1]+1]
                else:
                    for k in range(p_start[index], p_end[index + int(confidence_burr[index]/2)-1]+1):
                        pred[k] = pred[p_start[index]-1]
                    for k in range(p_end[index + int(confidence_burr[index])-1], p_end[index + int(confidence_burr[index]/2)]-1, -1):
                        pred[k] = pred[p_end[index + int(confidence_burr[index])-1]+1]
            index += confidence_burr[index] -1
        index += 1

    confidence_burr = []
    # locate the burrs
    p_label, p_start, p_end = get_segments(pred, id2class_map)
    for i in range(len(p_label)):
        if ((p_end[i] - p_start[i]) <= config.lbs_window ): 
            sum =[]
            for m in range(p_start[i],p_end[i]):
                sum.append(np.max(output[:,m:m+1]))
            if(np.mean(sum) <= config.lbs_Confidence):
                confidence_burr.append(1)
            else:
                confidence_burr.append(0)
        else:
            confidence_burr.append(0)
    # remove isolated burrs
    for i in range(len(p_label)):
        if(confidence_burr[i] == 1):
            if(i == 0):
                for k in range(p_end[i], p_start[i]-1, -1):
                    pred[k] = pred[k+1]               
            elif(i == (len(confidence_burr)-1)): 
                for k in range(p_start[i], p_end[i]+1):
                    pred[k] = pred[k-1]
            else:
                q = int((p_end[i-1]-p_start[i-1])/(p_end[i-1]-p_start[i-1]+p_end[i+1]-p_start[i+1])*(p_end[i]-p_start[i]))
                for k in range(p_start[i], p_start[i]+q+1):
                    pred[k] = pred[p_start[i]-1]
                for k in range(p_end[i], p_start[i]+q-1, -1):
                    pred[k] = pred[p_end[i]+1]

    return pred