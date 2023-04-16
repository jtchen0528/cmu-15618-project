#%%
import os
import json
dirname = "pyomp_static"
dirs = sorted(os.listdir(dirname))
datas = []
dirs = dirs[3:] + dirs[:3]
dirs
#%%
for dir in dirs:
    filepath = os.path.join(dirname, dir, "time_log.txt")
    f = open(filepath, 'r')
    data_json = json.loads(f.read())
    datas.append(dict(data_json))
# %%
len(datas)
# %%
pyomp_runtime = []
kmeans_runtime = []
total_runtime = []
init_runtime = []
for data in datas:
    local_kmeans_runtime = 0
    local_pyomp_runtime = 0
    local_total_runtime = 0
    for k, v in data.items():
        if k[:10] == 'iter_data_' and k != 'iter_data_0':
            local_kmeans_runtime += v
        if k == 'centroid_init':
            local_total_runtime += v
            init_runtime.append(v)
        if k == 'total':
            local_total_runtime += v
        if k[:5] == 'iter_' and k[:6] != 'iter_d':
            local_pyomp_runtime += v
    kmeans_runtime.append(local_kmeans_runtime)
    pyomp_runtime.append(local_pyomp_runtime)
    total_runtime.append(local_total_runtime)
# %%
kmeans_runtime
#%%
pyomp_runtime
# %%
total_runtime
#%%
init_runtime
# %%
update_runtime = []
for i in range(len(kmeans_runtime)):
    update_runtime.append(pyomp_runtime[i] - kmeans_runtime[i])
update_runtime
# %%
