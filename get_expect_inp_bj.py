import numpy as np
import h5py
import os


time_interval = ['01', '02', '03', '04', '05', '06', '07', '08',
                 '09', '10', '11', '12', '13', '14', '15', '16',
                 '17', '18', '19', '20', '21', '22', '23', '24',
                 '25', '26', '27', '28', '29', '30', '31', '32',
                 '33', '34', '35', '36', '37', '38', '39', '40',
                 '41', '42', '43', '44', '45', '46', '47', '48']

def filterExt():
    '''
    本函数是用来过滤所提供的外部环境数据
    1. 根据.h5文件或得.npy数据
    2. 根据.npy数据来过滤外部环境信息,获得外部环境因素.npy数据
        总共有59006条，但有流量数据的有21360条
    3. './MinMax/all_data.npy'是经过训练集均值方差归一化处理以后的数据,是在dataset_bj.py中获得
        其中'./MinMax/all_data.npy'是从dataset/dataset.py代码中获取的
    '''
    f = h5py.File('./TaxiBJ/BJ_Meteorology.h5', 'r')
    temperature = f['Temperature'][()]
    weather = f['Weather'][()]
    windspeed = f['WindSpeed'][()]
    ext_date = f['date'][()]
    all_date_array = np.load('./TaxiBJ/MinMax/normal_date.npy', allow_pickle=True)

    all_date = np.concatenate(list(all_date_array))

    # filter out data that is not in the dataset
    te_list = []
    we_list = []
    wi_list = []
    da_list = []

    for i, all_d in enumerate(all_date):
        all_d = all_d.decode()
        for j, ext_d in enumerate(ext_date):
            ext_d = ext_d.decode()
            if ext_d == all_d:
                te_list.append(temperature[j])
                we_list.append(weather[j])
                wi_list.append(windspeed[j])
                da_list.append(ext_date[j])
                break

    we_array = np.array(we_list)
    da_array = np.array(da_list)

    return we_array, da_array


def preExtCls(we_array, da_array):
    '''
    本函数是用来将天气和假期因素转为四种环境类型的heatmap数据，规则如下：
    1、先统计一下每天每时刻的天气类型,共17种
    2、考虑到人类活动的时间大致是早7到晚8，因此在这个时间段统计出一个众数来近似作为一天的天气类型
    3、针对每一种天气类型，对原数据进行统计，求得48个时刻序列的平均值
        对每一个时刻进行for循环，求序列的平均值，（此处只是用了训练集）
        当某种天气类型的天只有一天的时候，采用滑窗后面的序列会用到相邻下一天的数据
    '''
    # 21360
    data_array = np.load('./TaxiBJ/MinMax/normal_data.npy', allow_pickle=True)
    # (21360,2,32,32) MinMax
    all_data = np.concatenate(list(data_array))
    # (21360,) b'2013070101'
    all_date = da_array
    # (21360,17) [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    all_weather = we_array

    weather_data_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    weather_date_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    count_list = [0] * 17

    # 将one-hot向量转化为下标表示，并
    for i, _ in enumerate(all_weather[::48]):
        _, index = np.nonzero(all_weather[i * 48:i * 48 + 48])
        max_num = np.argmax(np.bincount(index[14:40])).astype(int)
        weather_data_list[max_num].append(all_data[i * 48:i * 48 + 48])
        weather_date_list[max_num].append(all_date[i * 48:i * 48 + 48])
        count_list[max_num] += 1

    ## 获取在全数据集中外部条件的分类日期，即给每一天分配一个类
    we_list = [[], []]
    for i, wa_li in enumerate(weather_date_list):
        if len(wa_li) != 0:
            if i == 0 or i == 1 or i == 2:
                # sunny
                for date in wa_li:
                    we_list[0] = we_list[0] + date.tolist()
            else:
                # no sunny
                for date in wa_li:
                    we_list[1] = we_list[1] + date.tolist()

    ## 获取在子数据集中节假日的分类日期，即确定假期和非假期的日期
    f = open('./TaxiBJ/BJ_Holiday.txt', 'r')
    holiday_date = [date.strip() for date in f.readlines()]

    # len=36
    holiday_data_list = []
    holiday_date_list = []
    not_holiday_data_list = []
    not_holiday_date_list = []
    for idx, date in enumerate(all_date[::48]):
        date = date.decode()
        if date[0:8] in holiday_date:
            holiday_data = all_data[idx * 48:idx * 48 + 48]
            holiday_data_list.append(holiday_data)
            holiday_date_list.append(all_date[idx * 48:idx * 48 + 48])
        else:
            not_holiday_data_list.append(all_data[idx * 48:idx * 48 + 48])
            not_holiday_date_list.append(all_date[idx * 48:idx * 48 + 48])

    ho_ex_data = [[], [], [], []]
    ho_ex_date = [[], [], [], []]
    for ho_date, ho_data in zip(holiday_date_list, holiday_data_list):
        if ho_date[0] in we_list[0]:
            # 7  28
            # print('今天是假期而且天气很好：\n', ho_date[0], we_list[0])
            ho_ex_date[0].append(ho_date)
            ho_ex_data[0].append(ho_data)
        else:
            # 6  8
            # print('今天是假期但是天气不好：\n', ho_date[0], we_list[1])
            ho_ex_date[1].append(ho_date)
            ho_ex_data[1].append(ho_data)

    for not_ho_date, not_ho_data in zip(not_holiday_date_list, not_holiday_data_list):
        if not_ho_date[0] in we_list[0]:
            # 69  292
            # print('今天不是假期而且天气很好：\n', not_ho_date[0], we_list[0])
            ho_ex_date[2].append(not_ho_date)
            ho_ex_data[2].append(not_ho_data)
        else:
            # 56  117
            # print('今天不是假期但是天气不好：\n', not_ho_date[0], we_list[1])
            ho_ex_date[3].append(not_ho_date)
            ho_ex_data[3].append(not_ho_data)

    return ho_ex_data, ho_ex_date


def getExtExp(date_array, ho_ex_data, ho_ex_date, save_path, LENGTH):
    """
    :param ho_ex_data: 四种类型数据的分组数据
    :param ho_ex_date: 四种类型日期的分组数据
    :param length: 控制序列长度
    :return: 四种外部因素类型的48个时刻npy数据及类型----Expectation input
    """
    all_date = date_array
    test_date_array = np.load('./TaxiBJ/Split_date/fetch_test_date.npy', allow_pickle=True)
    ## 求分类数据的平均
    all_data_av = []
    for cl_data, cl_date in zip(ho_ex_data, ho_ex_date):
        cl_data = np.concatenate(cl_data)
        cl_date = np.concatenate(cl_date)

        day_av_list = []
        for _, interval in enumerate(time_interval):
            total = np.zeros((LENGTH, 2, 32, 32))
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
    for sub_day in all_date[::48]:
        for i, ex_date in enumerate(ho_ex_date):
            ex_date = np.concatenate(ex_date).tolist()
            if sub_day in ex_date:
                external_avg.append(all_data_av[i])
                external_cls.append([i]*48)
                break

    all_data_con = np.concatenate(external_avg)
    all_cls_con = np.concatenate(external_cls)

    all_input_list = [all_data_con[0:4848], all_data_con[4848:9216], all_data_con[9216:14736],
                      all_data_con[14736:]]
    all_cls_list = [all_cls_con[0:4848], all_cls_con[4848:9216], all_cls_con[9216:14736],
                    all_cls_con[14736:]]

    all_input_array = np.array(all_input_list)
    all_cls_array = np.array(all_cls_list)

    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    np.save(save_path + 'expectation_cls.npy', all_cls_array)
    np.save(save_path + 'expectation_inp.npy', all_input_array)


if __name__ == '__main__':
    # 1. 过滤所给外部环境因素数据集
    we_array, da_array = filterExt()

    # 2. 预处理天气和假期因素,分为四种类型
    ho_ex_data, ho_ex_date = preExtCls(we_array, da_array)

    # 3. Expectation input
    save_path = './TaxiBJ/AVG6_4/'
    getExtExp(da_array, ho_ex_data, ho_ex_date, save_path, LENGTH=6)

