from math import pi
import random


class Solution:
    def search(self, nums, target: int) -> int:
        left, right= 0, len(nums)-1
        found = -1
        while left <= right and found==-1:
            mid = (left+right)//2
            if nums[mid] == target:
                found = mid

            else:
                if target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid -1
        return found
def getValue(gifts, n):
    count = 0
    ans = 0
    for i in gifts:
        if count==0:
            ans = i
        if ans == i:
            count += 1
        else:
            count -= 1
    return ans  




def quick_sort(nums,first,last):
        mid_value = nums[first]

        n = len(nums)
        low = first
        high = last
        
        while low < high:
            while low < high and nums[high] > mid_value:
                high -= 1
            nums[low] = nums[high]
            low += 1

            while low < high and nums[low] < mid_value:
                low += 1
            nums[high] = nums[low]
            high -= 1
        

        # 循环结束时
        nums[low] = mid_value

        # 对low左边执行快排
        quick_sort(first,low-1)

        # 对low的右边快排
        quick_sort(low+1,last)  

def partition(nums,left,right):
    i,j = left, right
    pivot = nums[left]
    while i != j:
        while i<j and nums[j] > pivot:
            j -= 1
        while i<j and nums[i] <= pivot:
            i += 1
        if i < j:
            nums[i],nums[j] = nums[j],nums[i]
    nums[i],nums[left] = nums[left], nums[i]
    return i

def quick(nums,left,right):
    def partition(nums,left,right):
        i,j = left, right
        pivot = nums[left]
        while i != j:
            while i<j and nums[j] > pivot:
                j -= 1
            while i<j and nums[i] <= pivot:
                i += 1
            if i < j:
                nums[i],nums[j] = nums[j],nums[i]
        nums[i],nums[left] = nums[left], nums[i]
        return i

    if left >= right:return
    pivot = partition(nums,left,right)
    quick(nums, left, pivot-1)
    quick(nums, pivot+1, right)

def quicksort(li, start, end):
    # 分治 一分为二
    # start=end ,证明要处理的数据只有一个
    # start>end ,证明右边没有数据
    if start >= end:
        return
    # 定义两个游标，分别指向0和末尾位置
    left = start
    right = end
    # 把0位置的数据，认为是中间值
    mid = li[left]
    while left != right:
        # 让右边游标往左移动，目的是找到小于mid的值，放到left游标位置
        while left < right and li[right] >= mid:
            right -= 1
        li[left] = li[right]
        # 让左边游标往右移动，目的是找到大于mid的值，放到right游标位置
        while left < right and li[left] < mid:
            left += 1
        li[right] = li[left]
    # while结束后，把mid放到中间位置，left=right
    li[left] = mid
    # 递归处理左边的数据
    quicksort(li, start, left-1)
    # 递归处理右边的数据
    quicksort(li, left+1, end)

if __name__ == '__main__':
    # solution = Solution()
    # res = solution.search(nums = [5], target = 5)
    # print(res)
    # res = getValue([1,2,3,3,5,3,3],7)
    res = [random.randint(0,20) for i in range(10)]
    print(res)
    # quick(res,0,9)
    quicksort(res,0,9)
    print(res)
    import torch
    
    