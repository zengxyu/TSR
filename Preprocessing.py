from PIL import Image
import os
import shutil
import xlrd
import csv
import random
import numpy as np

#train 文件格式转换器，将图片由ppm格式转为png格式
def trainFormatConverter(dir_from_path,dir_to_path):
    #目的文件夹，如果不存在，则创建文件夹；如果存在，删除原来的文件夹，再创建文件夹
    if (not os.path.exists(dir_to_path)):
        os.mkdir(dir_to_path)
    else:
        shutil.rmtree(dir_to_path)#先删除原来的目录
        os.mkdir(dir_to_path)#再创建一个新目录
    # 判断源文件夹是否存在
    if (os.path.exists(dir_from_path)):
        # 获取该目录下的所有文件夹目录，例：00000，00001，00002
        dir_from_children_names = os.listdir(dir_from_path)
        print("dir_children_names:"+str(dir_from_children_names))
        for dir_from_children_name in dir_from_children_names:
            print("开始处理该子目录，dir_children_names:" + str(dir_from_children_name))
            dir_from_children_path = os.path.join(dir_from_path,dir_from_children_name)
            dir_to_children_path = os.path.join(dir_to_path,dir_from_children_name)
            if not os.path.exists(dir_to_children_path):
                os.mkdir(dir_to_children_path)
            fileNames = os.listdir(dir_from_children_path)
            # print("fileNames:" + str(fileNames))
            for fileName in fileNames:# 得到该文件下所有目录的路径
                (shotName,suffix) = os.path.splitext(fileName)
                # print("shotName:"+shotName)
                # print("suffix:"+suffix)
                if suffix == ".ppm":#判断后缀为ppm则进行转换
                    file_from_path = os.path.join(dir_from_children_path, fileName)
                    file_to_path = os.path.join(dir_to_children_path,(shotName+".png"))
                    # print(file_to_path)
                    img = Image.open(file_from_path)
                    img.save(file_to_path)
                elif suffix == ".csv":
                    file_from_path = os.path.join(dir_from_children_path, fileName)
                    file_to_path = os.path.join(dir_to_children_path, fileName)
                    shutil.copy(file_from_path, file_to_path)
    else:
        print("dir_from_path不存在")

#测试集格式转换器
def testFormatConverter(dir_from_path,dir_to_path):
    # 目的文件夹，如果不存在，则创建文件夹；如果存在，删除原来的文件夹，再创建文件夹
    if (not os.path.exists(dir_to_path)):
        os.mkdir(dir_to_path)
    else:
        shutil.rmtree(dir_to_path)  # 先删除原来的目录
        os.mkdir(dir_to_path)  # 再创建一个新目录
    index = 0#技术
    # 判断源文件夹是否存在
    if (os.path.exists(dir_from_path)):
        # 获取该目录下的所有文件夹目录，例：00000，00001，00002
        fileNames = os.listdir(dir_from_path)
        # print("fileNames:" + str(fileNames))
        for fileName in fileNames:  # 得到该文件下所有目录的路径
            if index % 1000 == 0:
                print("第"+str(index)+"个文件，开始处理该文件，fileName:" + str(fileName))
            (shotName, suffix) = os.path.splitext(fileName)
            # print("shotName:"+shotName)
            # print("suffix:"+suffix)
            if suffix == ".ppm":  # 判断后缀为ppm则进行转换
                file_from_path = os.path.join(dir_from_path, fileName)
                file_to_path = os.path.join(dir_to_path, (shotName + ".png"))
                # print(file_to_path)
                img = Image.open(file_from_path)
                img.save(file_to_path)
            elif suffix == ".csv":
                file_from_path = os.path.join(dir_from_path, fileName)
                file_to_path = os.path.join(dir_to_path, fileName)
                shutil.copy(file_from_path, file_to_path)
            index=index+1
        print("文件个数："+str(len(fileNames)))
    else:
        print("dir_from_path不存在")
#读取excel表格
#返回一个字典{文件名:标签}
def readLabelsfromExcel(file_path):
    sheet_name = "GT-00000"
    colnameindex = 0#第一行数据是列名
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_name(sheet_name)  # 获得表格
    nrows = table.nrows  # 拿到总共行数
    colnames = table.row_values(colnameindex)  # 某一行数据 ['姓名', '用户名', '联系方式', '密码']
    dict = {}
    for rownum in range(1, nrows):  # 也就是从Excel第二行开始，第一行表头不算
        row = table.row_values(rownum)
        if row:
            dict[row[0]] = row[-1]  # 表头与数据对应
    return dict

#读取CSV表格
#返回一个字典{文件名：标签}
def readLabelsfromCSV(file_path):
    #获取文件夹号码
    # file_parent_name = file_path.split('\\')[-1][3:8]
    # print("file_parent_name:"+file_parent_name)
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]#需要这么读取
        dict = {}
        for row in rows:  #读取CSV的每一行
            row_arr = row[0].split(';')
            file_name = str(row_arr[0]).replace("ppm","png")#一行的row_arr[0]是文件名，将文件名的后缀ppm->png
            # file_path = file_parent_name+"/"+file_name
            dict[file_name] = row_arr[-1]
        return dict

#将traning_image中的训练数据存储为CSV格式的数据
#1.发现根本不用CSV文件，直接读文件夹就行了
#2.由文件名读取图片，并将图片resize为相同的大小
#3.将图片转为数组，存在6个CSV文件中，最终CSV文件的每一行的格式是  数组，label
def makeCSVFiles(dir_root_path,dir_to_path):
    dir_root_children_names = os.listdir(dir_root_path)
    dict = {}
    i=0
    for dir_root_children_name in dir_root_children_names:#这个i就是标签
        dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
        file_names = os.listdir(dir_root_children_path)
        for file_name in file_names:
            (shot_name,suffix) = os.path.splitext(file_name)
            if suffix =='.ppm':
                file_path = os.path.join(dir_root_children_path,file_name)
                dict[file_path]=i
        i=i+1
    path_list = list(dict)
    random.shuffle(path_list)#打乱
    if os.path.exists(dir_to_path):#如果存在，先删除后创建
        shutil.rmtree(dir_to_path)
        os.mkdir(dir_to_path)
    else:
        os.mkdir(dir_to_path)
    file = open(os.path.join(dir_to_path, ('mnist_batch0' + ".csv")), 'w', newline='')
    j = 0
    for path in path_list:
        im = Image.open(path)
        image = im.resize((48,48))
        arr = np.array(image)
        label = dict[path]
        #每遍历一个image，则添加一个label
        example = []
        for m in arr.tolist():
            for n in m:
                example.extend(n)
        example.append(label)
        writer = csv.writer(file)
        writer.writerow(example)
        if j % 7000 == 0:
            print("j:" + str(j))
            index = int(j / 7000)
            file.close()
            # 存入文件中
            file = open(os.path.join(dir_to_path, ('mnist_batch' + str(index) + ".csv")), 'w', newline='')
        j = j+1
    file.close()
    print("dict len:"+str(len(dict)))

#制作训练集的CSV file文件，每一行包括文件位置和对应的标签
def makeTrainCSV(dir_root_path,dir_to_path):
    #目的文件件，如果存在就删除，再创建；如果不存在就直接创建
    if os.path.exists(dir_to_path):
        shutil.rmtree(dir_to_path)
        os.makedirs(dir_to_path)
    else:
        os.makedirs(dir_to_path)
    dir_root_children_names = os.listdir(dir_root_path)#列出该根目录下的所有子目录
    dict_all_class = {}#每一个类别的dict,{path:label}
    i=0
    file_train = open(os.path.join(dir_to_path, ('train_data' + ".csv")), 'w', newline='')
    # file_test = open(os.path.join(dir_to_path, ('test_data' + ".csv")), 'w', newline='')
    for dir_root_children_name in dir_root_children_names:  # 这个i就是标签
        dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
        if os.path.isfile(dir_root_children_path):
            break
        file_names = os.listdir(dir_root_children_path)
        for file_name in file_names:
            (shot_name, suffix) = os.path.splitext(file_name)
            if suffix == '.png':
                file_path = os.path.join(dir_root_children_path, file_name)
                dict_all_class[file_path] = i
        i=i+1

    list_train_all_class = list(dict_all_class)    #每一个子类别由字典转为列表，列表中只有字典的Key,即路径
    random.shuffle(list_train_all_class)  #打乱
    for path_train_path in list_train_all_class:
        label = dict_all_class[path_train_path]
        example = []
        example.append(path_train_path)
        example.append(label)
        print("example:"+str(example))
        writer = csv.writer(file_train)
        writer.writerow(example)

    file_train.close()
    print("list_train_all_class:" + str(list_train_all_class))
    print("list_train_all_class len:"+str(len(list_train_all_class)))
#制作测试集的CSV file文件，每一行包括文件位置和对应的标签
def makeTestCSV(dir_root_path,dir_to_path):
    # 目的文件件，如果存在就删除，再创建；如果不存在就直接创建
    # if os.path.exists(dir_to_path):
    #     shutil.rmtree(dir_to_path)
    #     os.makedirs(dir_to_path)
    # else:
    #     os.makedirs(dir_to_path)

    csv_file_path = ""#CSV的文件路径
    file_names = os.listdir(dir_root_path)  # 列出该根目录下的所有子目录
    # 读取数据中的CSV文件，该文件有文件名和标签的对应关系
    #找到文件是.CSV后缀的
    for file_name in file_names:
        (shot_name, suffix) = os.path.splitext(file_name)
        if suffix=='.csv':
            csv_file_path = os.path.join(dir_root_path,file_name)
    dict = readLabelsfromCSV(csv_file_path)

    dict_all_class = {}  # 每一个类别的dict,{path:label}

    # file_train = open(os.path.join(dir_to_path, ('train_data' + ".csv")), 'w', newline='')
    file_test = open(os.path.join(dir_to_path, ('test_data' + ".csv")), 'w', newline='')
    for file_name in file_names:
        (shot_name, suffix) = os.path.splitext(file_name)
        if suffix == '.png':
            file_path = os.path.join(dir_root_path, file_name)
            file_name = file_path.split('\\')[-1]
            dict_all_class[file_path] = dict[file_name]

    list_test_all_class = list(dict_all_class)  # 每一个子类别由字典转为列表，列表中只有字典的Key,即路径
    random.shuffle(list_test_all_class)  # 打乱
    for path_test_path in list_test_all_class:
        label = dict_all_class[path_test_path]
        example = []
        example.append(path_test_path)
        example.append(label)
        print("example:" + str(example))
        writer = csv.writer(file_test)
        writer.writerow(example)
    file_test.close()
    print("list_test_all_class:" + str(list_test_all_class))
    print("list_test_all_class len:" + str(len(list_test_all_class)))

#=============转换图片格式ppm->png=============
# dir_father = "E:\研\数据集\交通标志识别\GTSRB_Final_Training_Images\GTSRB\Final_Training"
# dir_father = "E:\研\数据集\交通标志识别\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images"
# dir_from_path = "E:\研\数据集\交通标志识别\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images"
# dir_to_path = "E:\DataSet\TrafficSignRecognition\GTSRB_Final_Training_Images\GTSRB\Final_Test"
# testFormatConverter(dir_from_path,dir_to_path)
# ============================================
#=============读取Excel表格===================
# file_path = "E:\研\数据集\交通标志识别\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images_png\\00000\GT-00000.csv"
# print(readLabelsfromExcel(file_path=file_path))
#============================================
''
#=============读取CSV文件===================
# file_path = "E:\研\数据集\交通标志识别\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images_png\\00003\GT-00003.csv"
# print(readLabelsfromCSV(file_path=file_path))
#============================================

#制作训练集的CSV文件
dir_root_path = "E:\DataSet\TrafficSignRecognition\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images_png"
makeTrainCSV(dir_root_path=dir_root_path,dir_to_path="data3")
#制作测试集的CSV文件
dir_root_path = "E:\DataSet\TrafficSignRecognition\GTSRB_Final_Training_Images\GTSRB\Final_Test"
makeTestCSV(dir_root_path=dir_root_path,dir_to_path="data3")

