import pandas as pd
import networkx as nx

# 加载CSV数据
df = pd.read_csv(r'D:\GraduationProject\GTrace\dataset\dataset_b\raw\2022-04-13.csv')

# 构建图数据结构
G = nx.from_pandas_edgelist(df, source='parentSpanId', target='spanId', create_using=nx.DiGraph())

# 将有向图转换为无向图
undirected_G = G.to_undirected()

# 设置子图大小阈值
threshold = 250

# 提取所有超过阈值的子图
subgraph_data = []
for nodes in nx.connected_components(undirected_G):
    if len(nodes) > threshold:
        subgraph = G.subgraph(nodes)
        subgraph_nodes = list(subgraph.nodes)
        subgraph_data.append(df[df['spanId'].isin(subgraph_nodes)])

# # 打印子图数据
# # print(subgraph_data)
# for sub in subgraph_data:
#     print(sub)
#     print()
# print(len(subgraph_data))
