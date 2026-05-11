python - <<'PY'
import pickle
from pprint import pprint

pkl_path = "/raid/zengchaolv/shuaizhe_vavam/nuscenes_unzip/nuscenes_datafiles/nuscenes_val_data_cleaned.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("type:", type(data))

if hasattr(data, "__len__"):
    print("len:", len(data))

if isinstance(data, dict):
    print("dict keys:")
    pprint(list(data.keys())[:20])
    first_key = next(iter(data))
    sample = data[first_key]
    print("\nfirst key:", first_key)
elif isinstance(data, (list, tuple)):
    sample = data[0]
else:
    sample = data

print("\nsample type:", type(sample))

if isinstance(sample, dict):
    print("sample keys:")
    pprint(sample.keys())
    print("\nsample preview:")
    for k, v in sample.items():
        print(f"{k}: {type(v)}")
        s = str(v)
        print(s[:500])
        print("-" * 80)
else:
    pprint(sample)
PY