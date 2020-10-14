import csv

def data_grouping():
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

    with open('gwhd_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([i, value])
            i = i + 1


    
data_grouping()
