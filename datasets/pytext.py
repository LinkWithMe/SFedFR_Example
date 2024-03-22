
from pickle_dataset import PickleDataset

pdataset = PickleDataset(pickle_root="pickle_datasets", dataset_name="celeba")
# 创建对应的pickle文件形式的数据集
pdataset.create_pickle_dataset()
# 读取保存的pickle数据集并获取相应的数据集
train_dataset = pdataset.get_dataset_pickle(dataset_type="train", client_id=0)
test_dataset = pdataset.get_dataset_pickle(dataset_type="test", client_id=2)