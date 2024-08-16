import csv

train_personID_path = './train_personID.csv'
train_without_personID_path = './train_without_personID.csv'

train_personID = []
train_without_personID = []

# 读取 train_personID.csv 文件
with open(train_personID_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        if len(row) < 7:  # 假设每行应至少有7个字段
            print(f"Skipping row in train_personID.csv due to insufficient data: {row}")
            continue
        train_personID.append(row)

# 读取 train_without_personID.csv 文件
with open(train_without_personID_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        if len(row) < 7:  # 假设每行应至少有7个字段
            print(f"Skipping row in train_without_personID.csv due to insufficient data: {row}")
            continue
        train_without_personID.append(row)

dicts = []
for data in train_without_personID:
    isFind = False
    for temp_data in train_personID:
        try:
            # 属于同一个视频
            if int(data[0]) == int(temp_data[0]):
                # 属于同一张图片
                if int(data[1]) == int(temp_data[1]):
                    if abs(float(data[2])-float(temp_data[2])) < 0.005 and abs(float(data[3])-float(temp_data[3])) < 0.005 and abs(float(data[4])-float(temp_data[4])) < 0.005 and abs(float(data[5])-float(temp_data[5])) < 0.005:
                        dict = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], int(temp_data[6])-1]
                        dicts.append(dict)
                        isFind = True
                        break
        except IndexError:
            print(f"Skipping comparison due to insufficient data in either row: {data} or {temp_data}")
            continue
    if not isFind:
        dict = [data[0], data[1], data[2], data[3], data[4], data[5], data[6], -1]
        dicts.append(dict)

# 写入处理后的数据到 train_temp.csv
with open('./train_temp.csv', "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(dicts)
