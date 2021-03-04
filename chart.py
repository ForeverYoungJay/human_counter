from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pylab as pl
from datetime import datetime
from matplotlib.pyplot import MultipleLocator


#从CSV文件中读取数据
def data():
    import csv
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19,20,21]
    times = []
    days = []
    filename = 'meetingrecords.csv'
    with open(filename) as f:
        reader = csv.reader(f)

        for row in reader:
            day =  row[1].split(" ")[0]
            days.append(day)
        days = list(set(days))
        nums = np.zeros(shape=(len(days),len(hours)),dtype="int64")
        def get_list(date):
            return datetime.strptime(date, "%d/%m/%Y").timestamp()

        days = sorted(days, key=lambda date: get_list(date))


    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            num = int(row[2])
            day = row[1].split(" ")[0]
            hour = int(row[1].split(" ")[1].split(":")[0])
            dayi = days.index(day)
            try:
                houri = hours.index(hour)
                nums[dayi, houri] += num
            except ValueError:
                pass





    return days,hours,nums




def chart(days,hours,nums):
    # 绘图设置
    print(days,hours,nums)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')  # 三维坐标轴
    # X和Y的个数要相同
    X = np.array(hours)
    Y = np.arange(len(days))
    #Y = [int(d.split("/") for d in days)]
    Z = nums  # 生成16个随机整数
    # meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    Z = Z.ravel()
    # 设置柱子属性
    height = np.zeros_like(Z)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 2  # 柱子的长和宽
    # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    # ax.bar3d(X, Y, height, width, depth, Z,color=['r'])  # width, depth, height
    colors = plt.cm.jet(Z.flatten() / float(Z.max()))
    ax.bar3d(X, Y, height, width, depth, Z, color=colors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.ylabel(days,size=5)
    plt.yticks()
    x_major_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    plt.yticks(np.arange(len(days)),days,rotation=-40,size=4)
    plt.show()







def main():
    days, hours, nums = data()
    chart(days, hours, nums)

if __name__ == '__main__':
    main()





















