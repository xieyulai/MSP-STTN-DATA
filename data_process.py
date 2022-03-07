import argparse
import pdb

import numpy as np
import h5py
import os
import math
from utils.date_fetcher import DataFetcher
from utils.util import MinMaxNormalization


class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')
    print('*' * 10 + 'DEBUG' + '*' * 10)
    print(datapath)

    def __init__(self, filename, train_mode, cpt, portion=1.0, test_days=-1, datapath=datapath):
        self.dataset = filename
        self.datapath = datapath
        self.portion = portion
        self.cpt = cpt

        if self.dataset == 'TaxiBJ':
            self.datafolder = 'TaxiBJ/raw_data'
            self.dataname = [
                'BJ13_M32x32_T30_InOut.h5',
                'BJ14_M32x32_T30_InOut.h5',
                'BJ15_M32x32_T30_InOut.h5',
                'BJ16_M32x32_T30_InOut.h5'
            ]
            self.nb_flow = 2
            self.dim_h = 32
            self.dim_w = 32
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.m_factor = 1.

        elif self.dataset == 'BikeNYC':
            self.datafolder = 'BikeNYC/raw_data'
            self.dataname = ['NYC14_M16x8_T60_NewEnd.h5']
            self.nb_flow = 2
            self.dim_h = 16
            self.dim_w = 8
            self.T = 24
            test_days = 10 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 16 * 8 / 81)

        elif self.dataset == 'DENSITY':
            self.train_mode = train_mode
            self.nb_flow = 1
            self.dim_h = 200
            self.dim_w = 200
            self.T = 24
            test_days = 3 if test_days == -1 else test_days

            self.m_factor = 1.
        else:
            raise ValueError('Invalid dataset')

        self.len_test = test_days * self.T

    def get_raw_data(self):
        """
         data:
         np.array(n_sample * n_flow * height * width)
         ts:
         np.array(n_sample * length of timestamp string)
        """
        raw_data_list = list()
        raw_ts_list = list()
        print("  Dataset: ", self.datafolder)
        for filename in self.dataname:
            f = h5py.File(os.path.join(self.datapath, self.datafolder, filename), 'r')
            _raw_data = f['data'][()]
            _raw_ts = f['date'][()]
            f.close()

            raw_data_list.append(_raw_data)
            raw_ts_list.append(_raw_ts)
        # delete data over 2channels

        return raw_data_list, raw_ts_list

    @staticmethod
    def remove_incomplete_days(data, timestamps, t=48):
        print("before removing", len(data))
        # remove a certain day which has not 48 timestamps
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + t - 1 < len(timestamps) and int(timestamps[i + t - 1][8:]) == t:
                days.append(timestamps[i][:8])
                i += t
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        print("after removing", len(data))
        return data, timestamps

    def trainset_of(self, vec):
        return vec[:int(math.floor((len(vec) - self.len_test) * self.portion))]

    def testset_of(self, vec):
        return vec[-math.floor(self.len_test * self.portion):]

    def load_data(self):
        """
        return value:
            X_train & X_test: [XC, XP, XT, Xext]
            Y_train & Y_test: vector
        """
        if self.dataset == 'DENSITY':
            raw_data = np.load(f'./{self.dataset}/raw_data/data.npy')[:, np.newaxis, :, :]
            raw_date = np.load(f'./{self.dataset}/raw_data/date.npy')
            if self.train_mode == 'scheme_2':
                raw_data = raw_data[16 * 24:]
                raw_date = raw_date[16 * 24:]
            data_list = [raw_data]
            ts_new_list = [raw_date]
        else:
            print('Preprocessing: Reading HDF5 file(s)')
            raw_data_list, ts_list = self.get_raw_data()

            # filter dataset
            data_list, ts_new_list = [], []
            for idx in range(len(ts_list)):
                raw_data = raw_data_list[idx]
                ts = ts_list[idx]
                raw_data, ts = self.remove_incomplete_days(raw_data, ts, self.T)
                data_list.append(raw_data)  # 列表套列表套数组，最外层长度为4
                ts_new_list.append(ts)
            raw_data = np.concatenate(data_list)

        print('Preprocessing: Min max normalizing')
        mmn = MinMaxNormalization()
        train_dat = self.trainset_of(raw_data)
        mmn.fit(train_dat)
        new_data_list = [
            mmn.transform(data).astype('float32', copy=False)
            for data in data_list
        ]
        print('Context data min max normalizing processing finished!')

        # print(raw_data.shape)
        # print(len(new_data_list))
        data = np.array(new_data_list)
        date = np.array(ts_new_list)
        # print(data.shape)
        # print(date.shape)

        # ### 保存归一化数据
        if self.dataset == 'DENSITY':
            if self.train_mode == 'scheme_2':
                path_data = f'./{self.dataset}/MinMax_2'
            else:
                path_data = f'./{self.dataset}/MinMax_1'
        else:
            path_data = f'./{self.dataset}/MinMax'
        if os.path.exists(path_data):
            pass
        else:
            os.makedirs(path_data)
        np.save(path_data + '/normal_data.npy', data)
        np.save(path_data + '/normal_date.npy', date)

        ts_y_list = []
        for idx in range(len(ts_new_list)):
            ts_y = DataFetcher(new_data_list[idx], ts_new_list[idx], self.cpt, self.T).fetch_data()
            ts_y_list.append(ts_y)
        ts_y = np.concatenate(ts_y_list)

        train_date = self.trainset_of(ts_y)
        test_date = self.testset_of(ts_y)

        ## 保存过滤后的日期
        if self.dataset != 'DENSITY':
            path_date = f'./{self.dataset}/Split_date'
            if os.path.exists(path_date):
                pass
            else:
                os.makedirs(path_date)
            np.save(path_date+ '/train_date.npy', train_date)
            np.save(path_date+ '/test_date.npy', test_date)

    def parse_date(self, path_load, path_save, length):
        """
        :param path_load: './train_date.npy'  './test_date.npy'是经过选择close/period/trend后,得到的日期,在data_fetcher.py中获得
        :param path_save: './fetch_train_date.npy'  './fetch_test_date.npy' 是对日期数据惊醒解析,成为 '2013120101' 格式
        :param length: 代表数据的一天的序列长度
        :return:
        """
        date_array_load = np.load(path_load)

        l = date_array_load.shape[0]
        train_date_list = []
        for i in range(l // length):
            date = date_array_load[i * length:i * length + length]
            for j, dat in enumerate(date):
                j = str(j + 1).zfill(2)
                da = str(dat)
                date_str = da[:4] + da[5:7] + da[8:10] + j
                train_date_list.append(date_str)

        date_array_save = np.array(train_date_list)
        np.save(path_save, date_array_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass in some training parameters')
    parser.add_argument('--data_name', type=str, default='TaxiBJ', help='Process the name of the dataset')
    parser.add_argument('--T', type=int, default=48, help='The number of time slots in a day')
    parser.add_argument('--train_mode', type=str, default='scheme_1', help='Density dataset partition scheme selection')
    config = parser.parse_args()

    dataset = Dataset(filename=config.data_name, train_mode=config.train_mode, cpt=[4,1,1])
    dataset.load_data()
    if config.data_name != 'DENSITY':
        path = f'./{config.data_name}/Split_date/'
        dataset.parse_date(path + 'train_date.npy',path + 'fetch_train_date.npy',config.T)
        dataset.parse_date(path + 'test_date.npy',path + 'fetch_test_date.npy',config.T)
