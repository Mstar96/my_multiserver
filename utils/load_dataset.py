import json
import os
import random
from typing import List,Tuple,Dict,Any

def load_dataset(data_path:str):
    with open(data_path,"r") as f:
        data = json.load(f)
    metadata = data["dataset"]
    return metadata

def get_sample(dataset:set, index: int) -> Tuple[List[int], List[float]]:
        """
        获取指定索引的样本
        """
        sample = dataset[index]
        return sample["thread_list"], sample["fi_func"]

def get_random_sample(dataset:set) -> Tuple[List[int], List[float]]:
        """
        随机获取一个样本
        """
        index = random.randint(0, 999)
        return get_sample(dataset,index)

if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_path,'data','data')
    print(f"{project_path}\n{data_path}")
    metadata = load_dataset(data_path)
    print('metadata loaded')
    sample = get_sample(metadata,0)
    print(sample)
    randomsample = get_random_sample(metadata)
    print(randomsample)