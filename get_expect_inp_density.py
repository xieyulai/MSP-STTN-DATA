# /data/TaxiBJ/Pro/all_data.npy
import argparse

import numpy as np
import os


time_interval = ['01', '02', '03', '04', '05', '06', '07', '08',
                 '09', '10', '11', '12', '13', '14', '15', '16',
                 '17', '18', '19', '20', '21', '22', '23', '24']


def proWOHoliday(train_mode, ho_type):
    """
        获取工作日和非工作日，所对应的日期和map数据
        './MinMax/all_data.npy'是经过训练集均值方差归一化处理以后的数据,是在data_process.py中获得
    """
    if train_mode == 'scheme_2':
        all_data_array = np.load('./DENSITY/MinMax_2/normal_data.npy', allow_pickle=True)[0]
        all_date_array = np.load('./DENSITY/MinMax_2/normal_date.npy', allow_pickle=True)[0]
    else:
        all_data_array = np.load('./DENSITY/MinMax_1/normal_data.npy', allow_pickle=True)[0]
        all_date_array = np.load('./DENSITY/MinMax_1/normal_date.npy', allow_pickle=True)[0]

    if ho_type == 'ho':
        f = open('./DENSITY/Holiday.txt', 'r')
        holiday_date = [date.strip() for date in f.readlines()]
    else:
        f = open('./DENSITY/Holiday_Wd.txt', 'r')
        holiday_date = [date.strip() for date in f.readlines()]

    # len=36
    holiday_data_list = []
    holiday_date_list = []
    not_holiday_data_list = []
    not_holiday_date_list = []
    for idx, date in enumerate(all_date_array[::24]):
        date = date.decode()
        if date[0:8] in holiday_date:
            holiday_data = all_data_array[idx * 24:idx * 24 + 24]
            holiday_data_list.append(holiday_data)
            holiday_date_list.append(all_date_array[idx * 24:idx * 24 + 24])
        else:
            not_holiday_data_list.append(all_data_array[idx * 24:idx * 24 + 24])
            not_holiday_date_list.append(all_date_array[idx * 24:idx * 24 + 24])

    holiday_date_array = np.concatenate(holiday_date_list)
    not_holiday_date_array = np.concatenate(not_holiday_date_list)

    return holiday_date_array, not_holiday_date_array


def proExtCls(train_mode, holiday_date_array, not_holiday_date_array):
    """
        将天气和节假日的组合共分为四类，分别获取其对应的日期和数据
    """
    ho_date = holiday_date_array
    no_ho_date = not_holiday_date_array
    if train_mode == 'scheme_2':
        normal_date = np.load('./DENSITY/MinMax_2/normal_date.npy', allow_pickle=True)[0]
        normal_data = np.load('./DENSITY/MinMax_2/normal_data.npy', allow_pickle=True)[0]
    else:
        normal_date = np.load('./DENSITY/MinMax_1/normal_date.npy', allow_pickle=True)[0]
        normal_data = np.load('./DENSITY/MinMax_1/normal_data.npy', allow_pickle=True)[0]

    if train_mode == 'scheme_2':
        weather_list = []
        with open('./DENSITY/weather.txt', 'r') as f:
            for cls in f.readlines()[16:]:
                weather_list.append([int(cls.strip())] * 24)
            weather = np.concatenate(weather_list)
    else:
        weather_list = []
        with open('./DENSITY/weather.txt', 'r') as f:
            for cls in f.readlines():
                weather_list.append([int(cls.strip())] * 24)
            weather = np.concatenate(weather_list)

    ho_ex_data = [[], [], [], []]
    ho_ex_date = [[], [], [], []]
    i = 0
    for date, wea in zip(normal_date[::24], weather[::24]):
        if date in ho_date and wea == 0:
            # print('今天是假期且天气很好：\n', date, wea)
            ho_ex_date[0].append(normal_date[i * 24:i * 24 + 24])
            ho_ex_data[0].append(normal_data[i * 24:i * 24 + 24])
        elif date in ho_date and wea == 1:
            # print('今天是假期但是天气不好：\n', date, wea)
            ho_ex_date[1].append(normal_date[i * 24:i * 24 + 24])
            ho_ex_data[1].append(normal_data[i * 24:i * 24 + 24])
        elif date in no_ho_date and wea == 0:
            # print('今天是非假期且天气很好：\n', date, wea)
            ho_ex_date[2].append(normal_date[i * 24:i * 24 + 24])
            ho_ex_data[2].append(normal_data[i * 24:i * 24 + 24])
        elif date in no_ho_date and wea == 1:
            # print('今天是非假期且天气不好：\n', date, wea)
            ho_ex_date[3].append(normal_date[i * 24:i * 24 + 24])
            ho_ex_data[3].append(normal_data[i * 24:i * 24 + 24])
        i = i + 1

    for ho in ho_ex_data:
        print(len(ho))
    return ho_ex_data, ho_ex_date


def getExtExp(train_mode, ho_ex_data, ho_ex_date, LENGTH):
    """
        根据pro_ext_cls()函数的输出，获取类别和对应的一组平均map
    """
    if train_mode == 'scheme_2':
        normal_date = np.load('./DENSITY/MinMax_2/normal_date.npy', allow_pickle=True)[0]
    else:
        normal_date = np.load('./DENSITY/MinMax_1/normal_date.npy', allow_pickle=True)[0]
    # all_data = np.load('./COVID/MinMax/normal_data.npy', allow_pickle=True)
    test_date_array = normal_date[-3*24:]
    ## 求分类数据的平均
    all_data_av = []
    for cl_data, cl_date in zip(ho_ex_data, ho_ex_date):
        if len(cl_data) == 0:
            all_data_av.append([])
        else:
            cl_data = np.concatenate(cl_data)
            # print(cl_data.shape)
            cl_date = np.concatenate(cl_date)
            # print(cl_date.shape)

            day_av_list = []
            for _, interval in enumerate(time_interval):
                total = np.zeros((LENGTH, 1, 200, 200))
                count = 0
                for k, date in enumerate(cl_date):
                    date = date.decode()
                    if date[-2:] == interval and k < len(cl_data) - LENGTH + 1:
                        if date not in list(test_date_array):
                            total += cl_data[k:k + LENGTH]
                            count += 1
                average = total / count
                day_av_list.append(average)
            all_data_av.append(np.stack(day_av_list))

    # 生成训练集
    external_avg = []
    external_cls = []
    for sub_day in normal_date[::24]:
        for i, ex_date in enumerate(ho_ex_date):
            if len(ex_date) == 0:
                continue
            else:
                ex_date = np.concatenate(ex_date).tolist()
                if sub_day in ex_date:
                    external_avg.append(all_data_av[i])
                    external_cls.append([i]*24)
                    break

    all_data_con = np.concatenate(external_avg)
    all_cls_con = np.concatenate(external_cls)

    print(all_data_con.shape)
    print(all_cls_con.shape)

    if train_mode == 'scheme_1':
        save_path = f'./DENSITY/AVG6_4_1/'
    elif train_mode == 'scheme_2':
        save_path = f'./DENSITY/AVG6_4_2/'
    else:
        raise print('path error')

    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    np.save(save_path + 'expectation_cls.npy', all_cls_con)
    np.save(save_path + 'expectation_inp.npy', all_data_con)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass in some training parameters')
    parser.add_argument('--train_mode', type=str, default='scheme_1', help='Density dataset partition scheme selection')
    config = parser.parse_args()

    # 2. 获取weekend and weekday class
    ho_date, no_ho_date = proWOHoliday(train_mode=config.train_mode, ho_type='ho_wd')

    # 3. 将四种类型数据分组
    ho_ex_data, ho_ex_date = proExtCls(config.train_mode, ho_date, no_ho_date)

    # 4. 获得Expectation input
    getExtExp(config.train_mode, ho_ex_data, ho_ex_date, LENGTH=6)


