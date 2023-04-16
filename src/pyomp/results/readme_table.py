#%%
import os
import json

f = open("pyomp/kmeans_omp1_8threads_mnist_pca0_10cluster_100iter/cpu_freq.log")

data = dict(json.loads(f.read()))
data
#%%
row_str = ""
for i in range(1, 13):
    row_str += f' Core_{i} |'

with open("output_table.md", "w") as f:
    f.write("| Time |" + row_str + "\n")
    f.write("| ---- |  ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |  ------ |\n")

    t = 0
    for i in range(len(data['core_0'])):
        curr_str = f'| {t} |'
        for c in range(12):
            val = data[f'core_{c}'][i]
            curr_str += f' {val} |'
        t += 5
        f.write(curr_str + "\n")
# %%
