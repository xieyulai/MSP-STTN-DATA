# MSP-STTN
Code and data for the paper [Multi-Size Patched Spatial-Temporal Transformer Network for Short- and Long-Term Grid-based Crowd Flow Prediction]()

Please cite the following paper if you use this repository in your research.
```
Under construction
```
This repo is for *data preparation*, more information can be found in [MSP-STTN](https://github.com/xieyulai/MSP-STTN). 

## MSP-STTN-DATA

### Raw Data
- Find the original data **TaxiBJ** and **BikeNYC** from [ATFM](https://github.com/liulingbo918/ATFM).
- Find the original data **CrowdDensityBJ** from [An AI competition](https://www.datafountain.cn/competitions/428/datasets) or [there is a BAIDU PAN link](https://github.com/agave233/2020-CCF-Crowd-Flow-Prediction).


### TaxiBJ
Keep the `TaxiBJ/` directory like this:
```bash
TaxiBJ/
___ BJ_Holiday.txt
___ BJ_Meteorology.h5
___ raw_data
    ___ BJ13_M32x32_T30_InOut.h5
    ___ BJ14_M32x32_T30_InOut.h5
    ___ BJ15_M32x32_T30_InOut.h5
    ___ BJ16_M32x32_T30_InOut.h5
```


### BikeNYC
Keep the `BikeNYC/` directory like this:
```bash
BikeNYC/
___ Holiday.txt
___ raw_data
___ ___ NYC14_M16x8_T60_NewEnd.h5
___ weather.txt
___ Weekend.txt

```

### Processing CrowdDensityBJ
Keep the `GET_DENSITY/` directory like this:
```bash
GET_DENSITY/
___ csv_data
___ ___ shortstay_20200117_20200131.csv
___ ___ shortstay_20200201_20200215.csv
___ csv_to_density.py
___ raw_data
```
Run `csv_to_density.py` to get the processed files.
The GET_DENSITY/ directory will be like this:
```
GET_DENSITY/
___ csv_data
___ ___ csv_here
___ ___ shortstay_20200117_20200131.csv
___ ___ shortstay_20200201_20200215.csv
___ csv_to_density.py
___ raw_data
    ___ data.npy
    ___ date.npy
    ___ raw_data_her
```

### CrowdDensityBJ
Keep the `DENSITY/` directory like this:
```
DENSITY/
___ Holiday.txt
___ Holiday_Wd.txt
___ raw_data
___ ___ data.npy
___ ___ date.npy
___ weather.txt
```


## Data Preprocessing
1.**Get the normalized data**
- Get the normalized data of the TaxiBJ dataset and put it in TaxiBJ/MinMax
```bash
python data_process.py --data_name 'TaxiBJ' --T 48 
```
- Get the normalized data of the BikeNYC dataset and put it in BikeNYC/MinMax
```bash
python data_process.py --data_name 'BikeNYC' --T 24 
```
- Get the normalized data of the DENSITY dataset and put it in DENSITY/MinMax_1 --scheme_1
```bash
python data_process.py --data_name 'DENSITY' --T 24 --train_mode scheme_1
```
- Get the normalized data of the DENSITY dataset and put it in DENSITY/MinMax_2 --scheme_2
```bash
python data_process.py --data_name 'DENSITY' --T 24 --train_mode scheme_2
```

2.**Get the CSTE input data**
- Get the CSTE input data of the TaxiBJ dataset and put it in TaxiBJ/AVG6_4
```bash
python get_expect_inp_bj.py
```
- Get the CSTE input data of the BikeNYC dataset and put it in BikeNYC/AVG6_4
```bash
python get_expect_inp_nyc.py
```
- Get the CSTE input data of the DENSITY dataset and put it in DENSITY/AVG6_4_1 --scheme_1
```bash
python get_expect_inp_density.py --train_mode 'scheme_1'
```
- Get the CSTE input data of the DENSITY dataset and put it in DENSITY/AVG6_4_2 --scheme_2
```bash
python get_expect_inp_density.py --train_mode 'scheme_2'
```

## Result
### TaxiBJ
The `TaxiBJ/` will be like this:
```
TaxiBJ/
___ AVG6_4
___ ___ expectation_cls.npy
___ ___ expectation_inp.npy
___ BJ_Holiday.txt
___ BJ_Meteorology.h5
___ MinMax
___ ___ normal_data.npy
___ ___ normal_date.npy
___ raw_data
___ ___ BJ13_M32x32_T30_InOut.h5
___ ___ BJ14_M32x32_T30_InOut.h5
___ ___ BJ15_M32x32_T30_InOut.h5
___ ___ BJ16_M32x32_T30_InOut.h5
___ Split_date
    ___ fetch_test_date.npy
    ___ fetch_train_date.npy
    ___ test_date.npy
    ___ train_date.npy
```

### BikeNYC
Your `BikeNYC/` will be like this:
```
BikeNYC/
___ AVG6_4
___ ___ expectation_cls.npy
___ ___ expectation_inp.npy
___ Holiday.txt
___ MinMax
___ ___ normal_data.npy
___ ___ normal_date.npy
___ raw_data
___ ___ NYC14_M16x8_T60_NewEnd.h5
___ Split_date
___ ___ fetch_test_date.npy
___ ___ fetch_train_date.npy
___ ___ test_date.npy
___ ___ train_date.npy
___ weather.txt
___ Weekend.txt
```


### CrowdDensityBJ
Your `DENSITY/` will be like this:
```
DENSITY/
___ AVG6_4_1
___ ___ expectation_cls.npy
___ ___ expectation_inp.npy
___ AVG6_4_2
___ ___ expectation_cls.npy
___ ___ expectation_inp.npy
___ Holiday.txt
___ Holiday_Wd.txt
___ MinMax_1
___ ___ normal_data.npy
___ ___ normal_date.npy
___ MinMax_2
___ ___ normal_data.npy
___ ___ normal_date.npy
___ raw_data
___ ___ data.npy
___ ___ date.npy
___ weather.txt
```


## Direct download
The processed files can be downloaded from:
- [TaxiBJ](https://pan.baidu.com/s/1-CHABngCbQoRI4QcQETiLw)
    - pw: `nxe9`
- [BikeNYC](https://pan.baidu.com/s/1X78cSALLeJElNA5YjP4YAQ)
    - pw:`okun`
- [CrowdDensityBJ](https://pan.baidu.com/s/1n7NklHEnXfUQA86pM7ioIA)
    - pw:`hip7`
