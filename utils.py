import csv
import json
from pprint import pprint
import collections
import scipy.io
import glob

def data_grouping_GWHD():
    result = {}
    with open('train.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # This skips the first row of the CSV file.
        next(csvreader)
        for row in csvreader:
            if row[0] in result:
                result[row[0]] = result[row[0]] + 1
            else:
                result[row[0]] = 1

    # print(result)
    result_count = {}
    for key, value in result.items():
        if value in result_count:
            result_count[value] = result_count[value] + 1
        else:
            result_count[value] = 1
    result_count = collections.OrderedDict(sorted(result_count.items()))
    with open('gwhd_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([key, value])
            i = i + 1

def data_grouping_COCO():
    with open('panoptic_val2017.json') as f:
        data = json.load(f)
    pprint(len(data['annotations'][5]['segments_info']))

    result_count = {}
    for data in data['annotations']:
        if len(data['segments_info']) in result_count:
            result_count[len(data['segments_info'])] = result_count[len(data['segments_info'])] + 1
        else:
            result_count[len(data['segments_info'])] = 1
    result_count = collections.OrderedDict(sorted(result_count.items()))
    with open('coco_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([key, value])
            i = i + 1

def data_grouping_BSR():
    result_count = {}
    matsloc = glob.glob("BSRtrain200_fromBSD500/*.mat")
    for matloc in matsloc:
        mat = scipy.io.loadmat(matloc)
        # print(len(mat['groundTruth'][0]))
        if len(mat['groundTruth'][0]) in result_count:
            result_count[len(mat['groundTruth'][0])] = result_count[len(mat['groundTruth'][0])] + 1
        else:
            result_count[len(mat['groundTruth'][0])] = 1

    result_count = collections.OrderedDict(sorted(result_count.items()))
    with open('BSR_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([key, value])
            i = i + 1        

# data_grouping_COCO()    
# data_grouping_GWHD()
data_grouping_BSR()
