"""
作者:user
日期:2021年09月22日
"""
import csv
import json
import re

import data.calculate_distance as dis

message_dict = {"time_gap": [], "dist": 0, "lats": [], "driverID": 0, "weekID": 6, "states": [], "timeID": 0,
                "dateID": 3, "time": 0, "lngs": [], "dist_gap": []}

with open('train.csv', 'r') as f:
    reader = csv.reader(f)
    result = list(reader)
    for i in range(len(result) - 1):
        r = re.split('\[|\]|\,|\ ', result[i + 1][8])
        lngs = []
        lats = []

        """for j in r:
             if j=='':
                r.remove(j)"""
        print(r)

        numberOfPoints = 0
        time = 0.0
        time_gap = []
        dist_gap = [0.0]

        for j in range(len(r)):
            if r[j] != '':
                if j % 4 == 2:
                    lngs.append(float(r[j]))
                    numberOfPoints += 1
                    time_gap.append(time)
                    time += 15.0
                elif j % 4 == 3:
                    lats.append(float(r[j]))

        time = (numberOfPoints - 1)*15.0

        for j in range(len(lngs)-1):
            dist_gap.append(dis.Calcu_Dist(lats[j], lngs[j], lats[j+1], lngs[j+1]))

        message_dict["time_gap"] = time_gap
        message_dict["lats"] = lats
        message_dict["time"] = time
        message_dict["lngs"] = lngs
        message_dict["dist_gap"] = dist_gap
        # print(message_dict)

        with open('train_05', 'w') as file:
            json.dump(message_dict, file)
            file.write('\n')
