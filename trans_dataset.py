import pandas as pd
import numpy as np

# train.csv 파일 읽기
data = pd.read_csv('./dataset/GSF/train.csv', header=None)

# 첫 번째 열에 timestamp 추가 (0, 1, 2, ...)
data.insert(0, 'timestamp', np.arange(len(data)))

# 열 이름 지정 (timestamp와 feature_0, feature_1, ..., feature_N)
num_features = data.shape[1] - 1
columns = ['timestamp'] + [f'feature_{i}' for i in range(num_features)]
data.columns = columns

# 새로운 형식으로 저장
data.to_csv('./dataset/GSF/train_converted.csv', index=False)
