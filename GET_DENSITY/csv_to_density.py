import pandas as pd
import numpy as np
from pyheatmap.heatmap import HeatMap
import cv2
import matplotlib.pyplot as plt

grid_x_min = 115.994279229
grid_x_max = 116.747278635
grid_y_min = 39.681856275
grid_y_max = 40.188086241

def csv_to_grid_01():
    """
    :return: 20200117_20200131 csv to grid
    """
    df = pd.read_csv('./csv_data/shortstay_20200117_20200131.csv', sep='\t')
    df.columns = ['date', 'hour', 'grid_x', 'grid_y', 'index']

    grid_h = []
    date_h = []
    for d in range(20200117, 20200132):
        df_y = df[df.iloc[:, 0] == np.array(d, dtype=int)]

        for h in range(24):
            df_h = df_y[df_y['hour'] == np.array(h, dtype=int)]
            grid_x_h = df_h['grid_x']
            grid_y_h = df_h['grid_y']
            index_h = df_h['index']

            grid = np.zeros((200, 200))
            for x, y, index in zip(grid_x_h, grid_y_h, index_h):
                grid_x = int(round(((x - grid_x_min) / (grid_x_max - grid_x_min) * 199), 0))
                grid_y = int(round(((y - grid_y_min) / (grid_y_max - grid_y_min) * 199), 0))
                if grid[grid_x, grid_y] != 0:
                    grid[grid_x, grid_y] += index
                else:
                    grid[grid_x, grid_y] = index

            date = (str(d) + str(h+1).zfill(2)).encode('utf-8')
            date_h.append(date)
            grid_h.append(grid)
    date_h = np.stack(date_h, axis=0)
    grid_h = np.stack(grid_h, axis=0)
    print(date_h.shape)
    print(grid_h.shape)

    return date_h, grid_h


def csv_to_grid_02():
    """
    :return: 20200201_20200215 csv to grid
    """
    df = pd.read_csv('./csv_data/shortstay_20200201_20200215.csv', sep='\t')
    df.columns = ['date', 'hour', 'grid_x', 'grid_y', 'index']

    grid_h = []
    date_h = []
    for d in range(20200201, 20200216):
        df_y = df[df.iloc[:, 0] == np.array(d, dtype=int)]
        for h in range(24):

            df_h = df_y[df_y['hour'] == np.array(h, dtype=int)]
            grid_x_h = df_h['grid_x']
            grid_y_h = df_h['grid_y']
            index_h = df_h['index']

            grid = np.zeros((200, 200))
            for x, y, index in zip(grid_x_h, grid_y_h, index_h):
                grid_x = int(round(((x - grid_x_min) / (grid_x_max - grid_x_min) * 199), 0))
                grid_y = int(round(((y - grid_y_min) / (grid_y_max - grid_y_min) * 199), 0))
                if grid[grid_x, grid_y] != 0:
                    grid[grid_x, grid_y] += index
                else:
                    grid[grid_x, grid_y] = index

            date = (str(d) + str(h+1).zfill(2)).encode('utf-8')
            date_h.append(date)
            grid_h.append(grid)
    date_h = np.stack(date_h, axis=0)
    grid_h = np.stack(grid_h, axis=0)
    print(date_h.shape)
    print(grid_h.shape)

    return date_h, grid_h


def integrate_data(date_0117,data_0117,date_0201,data_0201):
    data = np.concatenate((data_0117,data_0201), axis=0)
    date = np.concatenate((date_0117,date_0201), axis=0)

    np.save('./raw_data/data.npy', data)
    np.save('./raw_data/date.npy', date)


if __name__ == '__main__':
    date_0117,data_0117 = csv_to_grid_01()
    date_0201,data_0201 = csv_to_grid_02()
    integrate_data(date_0117,data_0117,date_0201,data_0201)

