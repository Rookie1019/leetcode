# 所有的乱七八糟的测试

import re
import random
# import torch
# from torch._C import qint32
# import torch.nn as nn
# from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec
# import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import tqdm


def quick(nums,left,right):
    def pa(nums,left,right):
        i,j = left,right
        pivot = nums[left]
        while i != j:
            while i < j and nums[j] > pivot:
                j -= 1
            while i < j and nums[i] <= pivot:
                i += 1
            if i < j:
                nums[j], nums[i] = nums[i], nums[j]
        nums[i], nums[left] = nums[left], nums[i]
        return i

    if left >= right:return
    pivot = pa(nums,left,right)
    quick(nums,left,pivot-1)
    quick(nums,pivot+1,right)

def quicksort(nums,left,right):
    if left >= right:return
    i,j = left,right
    pivot = nums[left]
    while i != j:
        while i < j and nums[j] > pivot:
            j -= 1
        nums[i] = nums[j]
        while i < j and nums[i] <= pivot:
            i += 1
        nums[j] = nums[i]
    nums[i] = pivot

    quicksort(nums,left,i-1)
    quicksort(nums,i+1,right)


def merge(left,right):
    l, r = 0,0
    res = []
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            res.append(left[l])
            l += 1
        else:
            res.append(right[r])
            r += 1
    # res = res + left[l:] + right[r:]
    res.extend(left[l:])
    res.extend(right[r:])
    return res

def mergesort(nums):
    def merge(left,right):
        l, r = 0,0
        res = []
        while l < len(left) and r < len(right):
            if left[l] < right[r]:
                res.append(left[l])
                l += 1
            else:
                res.append(right[r])
                r += 1
        # res = res + left[l:] + right[r:]
        res.extend(left[l:])
        res.extend(right[r:])
        return res
    
    
    if len(nums) <= 1:return nums
    mid = len(nums) // 2
    left = mergesort(nums[:mid])
    right = mergesort(nums[mid:])

    return merge(left,right)
        

def ferbo(n):
    if n == 1 or n == 2:
        return 1
    return  ferbo(n-2) + ferbo(n-1)




if __name__ == '__main__':
    rs = [random.randint(0,20) for i in range(21)]
    # print(rs)
    # quicksort(rs,0,19)
    # print(rs)
    # print(7/2)
    # res = mergesort(rs)
    # print(res)
    res = ferbo(6)
    print(res)






