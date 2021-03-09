import math
import numpy as np
from collections import Counter



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

def minimumTotal(triangle) -> int: # 3.4
    # 120. 三角形最小路径和 medium
    n = len(triangle)
    if not triangle:return 0
    
    # 初始化状态矩阵
    dp = [[0]*i for i in range(1,n+1)]
    # 初始值
    dp[0][0] = triangle[0][0]
    for i in range(1,n):
        # 三角形的左边界
        dp[i][0] = dp[i-1][0]+triangle[i][0]
        for j in range(1, i):
            # 状态转移方程 
            dp[i][j] = min(dp[i-1][j],dp[i-1][j-1]) + triangle[i][j]
        # 三角形的右边界
        dp[i][i] = dp[i-1][i-1]+triangle[i][i]
            # if j == 0:
            #     dp[i][j] = dp[i-1][j]+triangle[i][j]
            # elif j == n-1:
            #     dp[i][j] = dp[i-1][j-1]+triangle[i][j]
            # else:
            #     dp[i][j] = dp[i-1][j] + dp[i-1][j-1]+triangle[i][j]
    print(dp)
    return min(dp[-1])  

    #另外一种优化的空间的解法  中还没弄明白
    # n = len(triangle)
    # f = [0] * n
    # f[0] = triangle[0][0]

    # for i in range(1, n):
    #     f[i] = f[i - 1] + triangle[i][i]
    #     for j in range(i - 1, 0, -1):
    #         f[j] = min(f[j - 1], f[j]) + triangle[i][j]
    #     f[0] += triangle[i][0]
    
    # return min(f)

def minPathSum(grid) -> int: # 3.4
    # 64. 最小路径和 medium
    n = len(grid)
    if not n:return 0
    


def ceshi():
    pass








if __name__ == '__main__':
    # obj = reverse(123)
    # obj = myPow(2.00000,10)
    # obj = lengthOfLongestSubstring('abcabcbb')
    # obj = countBits(5)

#------------------------------------------------------------
# 动态规划
    # obj = climbStairs(7)
    # obj = uniquePaths(3,7)
    # obj = maxEnvelopes([[5,4],[6,5],[6,7],[2,3]])
    # obj = lengthOfLIS([10,9,2,5,3,7,101,18])
    obj = minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]])
#------------------------------------------------------------

    print(obj)






    ceshi()