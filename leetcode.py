import os
from collections import Counter

def predictPartyVictory(senate:str):
    c = list(senate)
    if len(senate) == 0:
        return None
    elif len(senate) == 1 or len(c) == 2:
        return 
    else:
        a = []
        b = []
        for i in range(1,len(senate)):
            a.append(senate[i-1])
            if senate[i] == a[0]:
                b.append(senate[i-1])
                break
            else:
                a.remove(a[0])
        if b[0] == 'D':
            return 'Dire'
        else:
            return 'Randiant'

def a():
    c = os.getcwd()
    print(c)

def romanToInt(s: str) -> int:
    a = {'I': 1,
         'V': 5,
         'X': 10,
         'L': 50,
         'C': 100,
         'D': 500,
         'M': 1000
         }
    total = a[s[-1]]
    for i in reversed(range(len(s)-1)):
        if a[s[i]] < a[s[i+1]]:
            total -= a[s[i]]
        else:
            total += a[s[i]]
    return total


def majorityElement(nums):
    dic = {}
    le = len(nums)
    for i in range(le):
        if nums[i] not in dic:
            dic[nums[i]] = 1
        else:
            dic[nums[i]] += 1
    p = []
    for b in dic:
        # if dic[b] < le/2:
        #     continue
        if dic[b] > le/2:
            p.append(b)
        # else:
        #     return -1
    if len(p) > 0:
        return p[0]
    return -1


def molo(nums):
    a = [nums[0]]
    for i in range(1,len(nums)):
        if len(a) != 0:
            if nums[i] != a[0]:
                a.remove(a[0])
                nums.remove(nums[i])
        else:
            a.append(nums[i])
    print(a)


def minimumTotal(triangle):
    a = triangle[0][0]
    for i in range(len(triangle)):
        # for j in range(len(triangle[i])):
            # print(triangle[i][j])
        # print(i)
        continue
    print(len(triangle))


def threeConsecutiveOdds(arr):
    if len(arr) < 3:
        return False
    else:
        for i in range(1, len(arr)-1):
            if arr[i-1]%2==1 and arr[i]%2==1 and arr[i+1]%2==1:
                return True
        return False

c = []
def isHappy(n):
    if n == 1:
        return True
    s = str(n) # 转成字符串
    a = 0 # 和
    for i in range(len(s)):
        a += int(s[i]) * int(s[i])
    if a != 1:
        if a not in c:
            c.append(a)
            isHappy(a)
        else:
            return False
    else:
        return True



def missingNumber(nums):
    # for i,j in enumerate(nums):
    #     if i < j:
    #         return i


    # midpoint = len(nums) // 2
    # for i in range(len(nums)):
    #     if midpoint < nums[i]:
    #         return missingNumber(nums[:i])
    #     else:
    #         return missingNumber(nums[i+1:])

    left, right = 0, len(nums)-1
    while left <= right:
        mid = (left+right)//2
        if nums[mid] == mid:
            left = mid + 1
        else:
            right = mid - 1
    return left


def hIndex(citations):
    for i,j in enumerate(reversed(citations)):
        if j <= len(citations)-i:
            return j
    

def merge(intervals):
    '''
    56
    以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。
    '''
    intervals.sort(key=lambda x:x[0])
    a = []
    # for i in range(len(intervals)-1):
    #     if intervals[i][1] >= intervals[i+1][0]:
    #         a.append([intervals[i][0],intervals[i+1][1]])
    #     elif intervals[i][1] == intervals[i-1][1]:
    #         break
    #     else:
    #         a.append(intervals[i])
    # retu
    for interval in intervals:
        if not a or a[-1][1] < interval[0]:
            a.append(interval)
        else:
            a[-1][1] = max(interval[1],a[-1][1])
    return a


def longestOnes(A, K: int) -> int:
    
    res = 0 # 窗口值
    left, right = 0, 0  # 左右指针
    zeros = 0 # 0的个数
    while right < len(A):
        if A[right]==0:
            zeros += 1
        while zeros > K:
            if A[left] == 0:    
                zeros -= 1
            left += 1
        res = max(res, right-left+1)
        right += 1
    return res

def twoSum(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i+1,n):
            if nums[i] + nums[j] == target:
                return [i,j]
    return [] 

def kClosest(points,K):
    a = lambda x: x[0]**2 + x[1]**2
    points.sort(key=a)
    return points[:K]


def findShortestSubArray(nums):
    '''697 数组的度'''
    mp = dict()
    # for idx,item in enumerate(nums):
    #     if item in mp:
    #         mp[item] += 1
    #     else:
    #         mp[item] = 1
    # return mp
    for i, num in enumerate(nums):
        if num in mp:
            mp[num][0] += 1
            mp[num][2] = i
        else:
            mp[num] = [1, i, i]
    return mp

class NumArray: # 3.1
    # 303. 区域和检索 - 数组不可变  
    # 前缀和的问题 生成一个新的数组来统计给定数组的任意两个个数的和
    def __init__(self, nums):
        self.preSum = [0]
        for num in nums:
            self.preSum.append(self.preSum[-1] + num)

    def sumRange(self, i: int, j: int) -> int:
        return self.preSum[j+1] - self.preSum[i]


def reverse(x: int): # 3.1
    # 7. 整数反转
    r = False
    if x < 0:
        x = -x
        r = True
    nums = str(x)
    count = 0
    for i in range(len(nums),0,-1):
        count += int(nums[i-1])*10**(i-1)
    if (-2)**31 <= count <= 2**31 - 1:
        return -count if r else count
    return 0

def myPow(x: float, n: int): # 3.1   # 失败的 分治法
    # 50. Pow(x, n)
    if n == 0:return 1
    else:
        if n < 0:
            n = -n
            x = 1/x 
        if n % 2 == 0:
            return myPow(x,n/2) * myPow(x,n/2)
        else:
            return myPow(x,(n//2)/2) * myPow(x,(n//2)/2) *x
        return x

class NumMatrix: # 3.2
    # 304. 二维区域和检索 - 矩阵不可变 medium
    def __init__(self, matrix):
        self.new = [[0]]   
        # 这里优化成 self.new = [[0] for _ in range(len(matrix))]
        # 快速建立二维数组 就不用263行的append了 [0] 后面可以接*
        for i in range(len(matrix)):
            self.new.append([0])
            for j in matrix[i]:
                self.new[i].append(self.new[i][-1]+j)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        num = 0
        for i in range(row1,row2+1):
            num += self.new[i][col2+1] - self.new[i][col1]
        return num

def lengthOfLongestSubstring(s: str): # 3.2 失败
    # 3. 无重复字符的最长子串
    if not s:return 0
    pre =last= 0
    max_len = cur_len = 0
    num = []
    for i in range(len(s)):
        if s[i] not in num:
            num.append(s[i])
            last += 1
            max_len += 1
    return

def countBits(num: int): # 3.3 
    # 338. 比特位计数
    # nums = []
    # for i in range(num+1):
    #     nums.append(bin(i).count('1'))
    # return nums
    res = [0] * (num + 1)
    for i in range(1, num + 1):
        # print(res[i >> 1])
        res[i] = res[i >> 1] + (i & 1)
    return res

###########################################################################################################
def climbStairs(n: int): # 3.3 动态规划第一题
    # 70. 爬楼梯 easy
    dp = [0] * n
    dp[0] = 1
    dp[1] = 2
    for i in range(2,n):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n-1]

def uniquePaths(m: int, n: int) -> int: # 3.3 动态规划第二题
    # 62. 不同路径 medium
    # 初始化dp矩阵
    dp = np.zeros([m,n],dtype=int)
    # dp = [[0]*n for _ in range(m)] # m*n阶0矩阵 numpy里面有啥更好的方法是吗 
    # print(dp)
    for i in range(m):
        dp[i][0] = 1
    for i in range(n):
        dp[0][i] = 1 # 注意这里不是i 是1 因为不是说的步数 而是说的路线数

    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]

def maxEnvelopes(envelopes): # 3.4
    # 354. 俄罗斯套娃信封问题 hard
    '''二维的最长子序列问题 '''
    if len(envelopes) <= 1:return len(envelopes)
    # 用[][0]进行排序 如果[][0]相同 就排[][1]倒序
    envelopes.sort(key=lambda x: (x[0],-x[1])) 
    # print(envelopes)
    dp = [1] * len(envelopes) # 动态规划
    for i in range(len(envelopes)):
        for j in range(i):
            if envelopes[i][1] > envelopes[j][1]:
                dp[i] = max(dp[i],dp[j]+1)
    return max(dp)

def lengthOfLIS(nums): # 3.4
    # 300. 最长递增子序列 medium 
    if not nums:return 0
    dp = [1]*len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                # 状态转移表达式dp[i] = max(dp[i],dp[j]+1)
                dp[i] = max(dp[i],dp[j]+1)
    return max(dp)


def ceshi():
    pass


if __name__ == "__main__":
    import numpy as np
    # print(np.zeros([3,7],dtype=int))
    # x = predictPartyVictory('RDRDR')
    # print(x)
    # a()
    # a = romanToInt('MCMXCIV')

    # a = majorityElement([1,2,3])
    # molo([1,2,5,9,5,9,5,5,5])
#     minimumTotal([
#      [2],
#     [3,4],
#    [6,5,7],
#   [4,1,8,3]
# ])
    # a = threeConsecutiveOdds([1,2,34,3,4,5,7,23,12])
    # a = isHappy(7)
    
    # print(a)
    # a = missingNumber([0,1,2,3,4,6,7,8,9])
    # a = hIndex([0,1,3,5,6])
    # a = merge([[1,3],[2,6],[8,10],[15,18]])
    # a = longestOnes([1,1,1,0,0,0,1,1,1,1,0],2)
    # a = twoSum([15,11,7,2],9)
    # a = kClosest([[3,3],[5,-1],[-2,4]],2)
    # a = findShortestSubArray([1, 2, 2, 3, 1])
    # obj = reverse(123)
    # obj = myPow(2.00000,10)
    # obj = lengthOfLongestSubstring('abcabcbb')
    # obj = countBits(5)

#------------------------------------------------------------
# 动态规划
    # obj = climbStairs(7)
    # obj = uniquePaths(3,7)
    obj = maxEnvelopes([[5,4],[6,5],[6,7],[2,3]])
    # obj = lengthOfLIS([10,9,2,5,3,7,101,18])
#------------------------------------------------------------

    print(obj)






    ceshi()
    
    
    

