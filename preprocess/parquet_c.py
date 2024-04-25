

import pyarrow.parquet as pq

# 读取Parquet文件
table = pq.read_table('test-00000-of-00001.parquet')

# 查看表的schema
print(table.schema)

# 遍历所有行数据
for batch in table.to_batches():
    for row in batch:
        print(row)