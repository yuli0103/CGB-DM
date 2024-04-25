

import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('../Dataset/cgl/csv/train.csv')

# 删除 cls_elem 列值为 4 的行
df = df[df['cls_elem'] != 4]

# 修改 cls_elem 列的值
df['cls_elem'] = df['cls_elem'].apply(lambda x: 2 if x == 1 else 1 if x == 2 else x)

# 创建一个新的 CSV 文件
df.to_csv('../Dataset/cgl/csv/train_2.csv', index=False)