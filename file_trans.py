import numpy as np
import pandas as pd

# .npy 파일 로드
data = np.load('./dataset/WADI/train.npy')
data = np.load('./dataset/WADI/test_label.npy')
data = np.load('./dataset/WADI/test.npy')
# .csv 파일로 저장
pd.DataFrame(data).to_csv('./dataset/WADI/train.csv', index=False)
pd.DataFrame(data).to_csv('./dataset/WADI/test_label.csv', index=False)
pd.DataFrame(data).to_csv('./dataset/WADI/test.csv', index=False)

