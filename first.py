import torch
import numpy as np
import torch.nn as nn

class A(nn.Module):
    def __init__(self,nnd):
        self.nnd = 1


a = np.array([1,2,3])
c = torch.tensor([1.2,2.1,3.0])
# print(a)
b = torch.tensor(a)
# print('b',b)
lss = torch.nn.CTCLoss(a,c)
print('dasdafsf',lss)


# class ListNode:
# #     def __init__(self, val=0, next=None):
# #         self.val = val
# #         self.next = next

# class Solution:

#     def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
import os
import re
import torch
# import config
# from config import ws
# from duqu import ws
from torch.utils.data import DataLoader, Dataset


train_data_path = '/home/deep/workspace/Rookie/Personal/data/aclImdb/train'
test_data_path = '/home/deep/workspace/Rookie/Personal/data/aclImdb/test'




class My_Data(Dataset):
    def __init__(self,train=True):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        data_path = self.train_data_path if train else self.test_data_path

        temp_data_path = [os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]
        self.total_data_path = []
        for data_path in temp_data_path:
            # file_name = os.listdir(data_path)
    
            file_path = [os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith('.txt')]
            self.total_data_path.extend(file_path)


    def __getitem__(self,indics):
        temp_label = self.total_data_path[indics].split('/')[-2]
        label = 0 if temp_label=='neg' else 1

        with open(self.total_data_path[indics]) as f:
            content = f.readline()
            content = tokenlize(content)
            content = [i.lower().strip('.') for i in content]

            # content = ws.transform(content)

            # content = torch.LongTensor(content)
            # label = torch.LongTensor(label)
        
        return content,label

    def __len__(self):
        return len(self.total_data_path)

def tokenlize(content):
    content = re.sub('<.*?>',' ',content)
    filters = ['/t','/n','/x97','/96','#','$','%','&']
    content = re.sub('|'.join(filters),' ',content)
    tokens = [i.strip() for i in content.split()]
    return tokens

def get_dataloader(train=True,batch_size=config.batch_size):
    data = My_Data(train)
    data_loader = DataLoader(data,batch_size=batch_size,
                            shuffle=True,collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    '''
    batch: ([token label],[token label])
    '''
    content, label = list(zip(*batch))
    content = [ws.transform(i) for i in content]

    content = torch.LongTensor(content)
    label = torch.LongTensor(label)

    return content, label


if __name__ == "__main__":
    # a = My_Data()
    # print(a[10])
    for idx,(content,label) in enumerate(get_dataloader()):
        print(idx)
        print(content)
        print(label)
        break