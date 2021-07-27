import torch 
from torchvision.datasets import MNIST

# mnist = MNIST(r'/data',train=True,download=True)




def minz(num):
    m,n = len(num),len(num[0])
    if m==1 or n==1:return sum(num)

    dp = [[0]*n for _ in range(m)]
    dp[0][0] = num[0][0]
    for i in range(m):
        dp[i][0] = dp[i-1][0] + num[i][0]
    for i in range(n):
        dp[0][i] = dp[0][i-1] + num[0][i]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = min(dp[i-1][j],dp[i][j-1]) + num[i][j]
    return dp[-1][-1]        

a = minz([[1,3,1],[1,5,1],[4,2,1]])
print('a,',a)


# c = lambda x : x**2
# # print(c(6))

# def com(s:str):
#     nums = s.split(' ')
#     a = {}
#     for num in nums:
#         if num in a:
#             a[num] += 1
#         else:
#             a[num] = 1

# def aaa(s):
#     a = s.split(' ')
#     # print(type(s))
#     # a = []
#     # for i in s:
#     #     a.append(a)
#     print(a)
#     d = range(5)
#     print('range',d)

# aaa('dasda dsaffd gfgf,dsad')

# def pre(nums):
#     print(nums)
#     preSum = [0]
#     for num in nums:
#         preSum.append(preSum[-1] + num)
#     print(preSum)
#     print(preSum[3]-preSum[0]) # 0,2
#     print(preSum[6]-preSum[2]) # 2,5

#     print(preSum[6]-preSum[0]) # 0,5
    
# def preSum_mat(matrix):
#     # print(len(matrix))
#     new = [[0] for _ in range(len(matrix))]
#     # for i in range(len(matrix)):
#     #     new.append([0])
#     print(new)
    
#     for i in range(len(matrix)):
#         # new.append([0])
#         for j in matrix[i]:
#             new[i].append(new[i][-1]+j)
            

#     print('new',new)
#     D = [ [0] * (5 + 1) for _ in range(7 + 1)]
    
#     print('D',D)

# # pre([-2, 0, 3, -5, 2, -1])
# preSum_mat([
#   [3, 0, 1, 4, 2],
#   [5, 6, 3, 2, 1],
#   [1, 2, 0, 1, 5],
#   [4, 1, 0, 1, 7],
#   [1, 0, 3, 0, 5]
# ])
def binary_search(target,nums):
    left = 0
    right = len(nums) - 1
    found = False

    while left <= right and not found:
        mid = (left + right) // 2
        if target == nums[mid]:
            found = True
        else:
            if target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
    return found

def bubble(nums):

    n = len(nums)
    for i in range(n-1,0,-1):
        for j in range(i):
            if nums[j] > nums[j+1]:
                nums[j],nums[j+1] = nums[j+1],nums[j]
            else:
                continue
    return nums

def select(nums):
    n = len(nums)
    for i in range(n-1,0,-1):
        temp = 0
        for j in range(i+1):
            if temp < nums[j]:
                temp = nums[j]
                z = j
        nums[z] = nums[i]
        nums[i] = temp
    return nums

def insert(nums):
    n = len(nums)
    for i in range(1,n):
        j = i
        while j > 0:
            if nums[j] < nums[j-1]:
                nums[j],nums[j-1] = nums[j-1],nums[j]
                j -= 1
            else:
                break
    return nums
def quick_sort(li, start, end):
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
    quick_sort(li, start, left-1)
    # 递归处理右边的数据
    quick_sort(li, left+1, end)



if __name__ == '__main__':
    # a = binary_search(3,[1,2,3,4,5,6,7,8])
    # a = bubble([1,4,2,3,5,7,10,6,9,7])
    # a = select([1,4,2,3,5,7,10,6,9,7])
    # a = insert([1,4,2,3,5,7,10,6,9,7])
    a = [1,4,2,3,5,7,10,6,9,7]
    print(a)
    quick_sort(a,0,9)
    print(a)


