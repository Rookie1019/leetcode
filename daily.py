
import functools
import math
import re
import numpy as np
import collections
from collections import Counter, defaultdict
from functools import cmp_to_key




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
    m = len(grid)
    n = len(grid[0])
    if not grid:return 0
    if m==1 and n==1:
        return grid[0][0]
    # 初始化状态转移矩阵
    dp = [[0]*n for _ in range(m)]
    # 初始值
    dp[0][0] = grid[0][0]
    for i in range(m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for i in range(n):
        dp[0][i] = dp[0][i-1] + grid[0][i]
    
    for i in range(1,m):
        for j in range(1,n):
            # 状态转移矩阵
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    return dp[-1][-1]

    # 还没看 空间优化好的
    # dp = [float('inf')] * (len(grid[0])+1)
    # dp[1] = 0
    # for row in grid:
    #     for idx, num in enumerate(row):
    #         dp[idx + 1] = min(dp[idx], dp[idx + 1]) + num
    # return dp[-1]





class MyQueue: # 3.5
    # 232. 用栈实现队列
    def __init__(self):
        self.in_ = []
        self.out = []
       
    def push(self, x: int) -> None:
        while self.out:
            self.in_.append(self.out.pop())
        self.in_.append(x)
    def pop(self) -> int:
        if not self.out:
            while self.in_:
                self.out.append(self.in_.pop())
        return self.out.pop()
     
    def peek(self) -> int:
        if not self.out:
            while self.in_:
                self.out.append(self.in_.pop())
        return self.out[-1]


    def empty(self) -> bool:
        if len(self.in_)==0 and len(self.out)==0:
            return True
        return False

def minCut(s: str) -> int: # 3.8
    # 132. 分割回文串 II :
    # 给你一个字符串s，请你将s分割成一些子串，使每个子串都是回文。返回符合要求的最少分割次数
    # s = "aab" return : 1 只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
    if len(s) <= 1:return 0
    dp = []


    pass

def removeDuplicates(S: str): # 3.9
    # 1047. 删除字符串中的所有相邻重复项 easy
    if len(S) == 1:return S
    if len(S) == 2:
        if S[0] == S[1]:
            return S[0]
        else:
            return S
    a = [S[0]]
    for idx in range(1,len(S)):
        if a and S[idx] == a[-1] :
            a.pop()
        else:
            a.append(S[idx])
    

    return ''.join(a)

# def longestPalindrome(s: str): # 3.9 dp
#     # 5. 最长回文子串 medium
#     dp = [1]*len(s)
    
#     for 

# def calculate(s: str):
#     # fuhao = ['+','-','*','/']
#     in_d = []
#     out_d = []
    
#     for i in range(1,len(s)):

#     a = eval()

def isValidSerialization(preorder: str) -> bool: # 3.12
    # 331. 验证二叉树的前序序列化 medium
    stack = list()
    for i in preorder.split(','):
        stack.append(i)
        while len(stack) > 2 and stack[-1] == '#' and stack[-2] == '#' and stack[-3].isdigit():
            stack.pop()
            stack.pop()
            stack.pop()
            stack.append('#')
    return True if stack == ['#'] else False

    # num_slot = 1
    # for s in preorder.split(','):
    #     if num_slot == 0:
    #         return False

    #     if s == '#':
    #         num_slot -= 1
    #     else:
    #         # 遇到数字，空槽位减去一个再加上两个
    #         num_slot = num_slot - 1 + 2
    # return num_slot == 0
    # if preorder[0] == '#':return False
    # pre = preorder.split(',')
    # a = 0
    # b = 0
    # for i in pre:
    #     if i.isdigit():
    #         a += 1
    #     else:
    #         b += 1
    # return b-a==1

def nextGreaterElements(nums): # 3.6
    # 503. 下一个更大元素 II medium 
    stack, nums_len = list(), len(nums)
    res = [-1] * nums_len
    for i in range(nums_len*2):
        while stack and nums[stack[-1]] < nums[i%nums_len]:
            res[stack.pop()] = nums[i%nums_len]   
        stack.append(i%nums_len)
        
    return res

def nextGreaterElement(nums1,nums2):
    # 496. 下一个更大元素 I easy
    # stack = []
    # n = nums2
    # for i in nums1:
    #     while stack and stack[-1] < i:
    #         stack.pop()
    #     stack.append(i)
    # return stack    实现单调栈
    # stack = []
    # res_dict = {i:-1 for i in nums2}
    # for i in nums2:  # [4,1,2],[1,3,4,2]
    #     while stack and i > stack[-1]:
    #         small = stack.pop()
    #         res_dict[small] = i
    #     stack.append(i)
    # res = []

    # for j in nums1:
    #     res.append(res_dict[j])
    # return res
            
    res_dict = {i:-1 for i in nums2}
    stack = []
    for i in nums2:
        while stack and stack[-1] < i:
            small = stack.pop()
            res_dict[small] = i
        stack.append(i)
    
    res = []
    for i in nums1:
        res.append(res_dict[i])

    return res

class MyHashSet: # 3.13 星期六
    # 705. 设计哈希集合 easy
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []

    def add(self, key: int) -> None:
        self.data.append(key)

    def remove(self, key: int) -> None:
        b = []
        for i in self.data:
            if i != key:
                b.append(i)
        self.data = b

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        return key in self.data

class MyHashMap: # 3.14
    # 706. 设计哈希映射 easy

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        found = False
        for i in range(len(self.data)):
            if key == self.data[i][0]:
                self.data[i][1] = value
                found = True
                break
        if found == False:self.data.append([key,value])
            
    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        for i in range(len(self.data)):
            if key == self.data[i][0]:
                return self.data[i][1]
        return -1

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        for i in range(len(self.data)):
            if key == self.data[i][0]:
                self.data.remove([self.data[i][0],self.data[i][1]])
                break




def spiralOrder(matrix): # 3.15 
    # 54. 螺旋矩阵 medium
    res = []
    while matrix:
        res += matrix.pop(0)
        a = zip(*matrix)
        b = list(a)
        c = b[::-1]

        matrix = c
    return res

def generateMatrix(n): # 3.16
    # 59. 螺旋矩阵 II medium
    if n == 1:return [[1]] # 这句其实没什么用
    num = 1
    # 定义矩阵
    res = [[0] * n for _ in range(n)]
    # 定义左右上下边界值 注意边界值的考虑
    left, right, top, bottom = 0, n-1, 0, n-1
    while left <= right and top <= bottom:
        for col in range(left, right + 1):
            res[top][col] = num
            num += 1
        for row in range(top + 1, bottom + 1):
            res[row][right] = num
            num += 1
        if top < bottom and left < right:
            for col in range(right - 1, left, -1):
                res[bottom][col] = num
                num += 1
            for row in range(bottom ,top , -1):
                res[row][left] = num
                num += 1
        left += 1
        right -= 1
        top += 1
        bottom -= 1

    return res

def numDistinct(s:str, t:str): # 3.17 一看就dp 还是要想
    # 115. 不同的子序列 hard
    '''
    我来解释下2个问题，
    1： 为啥状态方程这样对？ 2：怎么想到这样的状态方程？
    我个人习惯dp[i][j] 表示为s[0-i] 和t[0-j]均闭区间的子序列个数，但这样不能表示s和t空串的情况
    所以声明 int[][] dp = new int[m + 1][n + 1]; 这样dp[0][x]可以表示s为空串，dp[x][0]同理。
    先不扣初始化的细节，假设dp[i][j] 就是s[i] 和t[j] 索引的元素子序列数量
    1：为啥状态方程是： s[i] == t[j] 时 dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
    s[i] != t[j] 时 dp[i][j] = dp[i-1][j]
    先看s[i] == t[j] 时，以s = "rara" t = "ra" 为例，当i = 3, j = 1时，s[i] == t[j]。
    此时分为2种情况，s串用最后一位的a + 不用最后一位的a。
    如果用s串最后一位的a,那么t串最后一位的a也被消耗掉，此时的子序列其实=dp[i-1][j-1]
    如果不用s串最后一位的a，那就得看"rar"里面是否有"ra"子序列的了，就是dp[i-1][j]
    所以 dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
    再看s[i] != t[j] 比如 s = "rarb" t = "ra" 还是当i = 3, j = 1时，s[i] != t[j]
    此时显然最后的b想用也用不上啊。所以只能指望前面的"rar"里面是否有能匹配"ra"的
    所以此时dp[i][j] = dp[i-1][j]
    2: 怎么想到这样状态方程的？
    一点个人经验，见过的很多2个串的题，大部分都是dp[i][j] 分别表示s串[0...i] 和t串[0...j]怎么怎么样 然后都是观察s[i]和t[j]分等或者不等的情况 而且方程通常就是 dp[i-1][j-1] 要么+ 要么 || dp[i-1][j]类似的
    类似的题比如有 10：正则表达式匹配 44：通配符匹配 编辑距离 1143：最长公共子序列等等的 还有几道想不起来了
    '''
    m, n = len(s), len(t)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][n] = 1
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            if s[i] == t[j]:
                dp[i][j] = dp[i+1][j+1] + dp[i+1][j]
            else:
                dp[i][j] = dp[i+1][j]
            
    return dp[0][0]

# class ListNode: 3.18
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
    
    
#     def reverseBetween(self, head: ListNode, left,right) -> ListNode:
#         pass

def longestPalindrome(s: str) -> str:
    # if len(s) in (1,2):return s[0]
    # dp = [i for i in range(len(s))]

    # for i in range(len(s)):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    ans = ""
    # 枚举子串的长度 l+1
    for x in range(n):
        # 枚举子串的起始位置 i，这样可以通过 j=i+l 得到子串的结束位置
        for i in range(n):
            j = i + x
            if j >= len(s):
                break
            if x == 0:
                dp[i][j] = True
            elif x == 1:
                dp[i][j] = (s[i] == s[j])
            else:
                dp[i][j] = (dp[i + 1][j - 1] and s[i] == s[j])
            if dp[i][j] and x + 1 > len(ans):
                ans = s[i:j+1]
    return ans

def hammingWeight(n: int) -> int: # 3.22
    # 191. 位1的个数 easy 只能调库
    n = bin(n)
    return n.count('1')

# def    3.23
# 341. 扁平化嵌套列表迭代器  medium
# 相当于遍历多叉树 学了一遍DFS
'''
class NestedInteger:
   def isInteger(self) -> bool:
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       """

   def getInteger(self) -> int:
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       """

   def getList(self) -> [NestedInteger]:
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """

class NestedIterator:
                    
    def __init__(self, nestedList):
        # self.nestedList = nestedList
        self.queue = collections.deque()
        self.dfs(nestedList)
        

    def dfs(self,nests):
        for nest in nests:
            if nest.isInteger():
                self.queue.append(nest.getInteger())
            else:
                self.dfs(nest.getList())

    def next(self):
        return self.queue.popleft()
        

    def hasNext(self):
        return len(self.queue)

'''

def find132pattern(nums) -> bool:
    if len(nums) <= 2:return False
    n = len(nums)
    # res = []
    for i in range(n-2):
        for j in range(i+1,n-1):
            if nums[i] < nums[j]:
                for x in range(j+1,n):
                    if nums[x] > nums[i] and nums[x] < nums[j]:
                        return True
    return False

# 3.28的题又是class 没办法写上来


def subsetsWithDup(nums): # 3.31
    # 90. 子集 II medium
    if nums == []:return nums
    res = [[]]
    for i in range(len(nums)):
        temp = []
        for j in range(i,len(nums)):
            temp.append(nums[j])
            if temp not in res:
                res.append(temp)
    return res
    # res = [[]]
    # for i in nums:
    #     temp = []
    #     for j in res:
    #         temp1 = j + [i]
    #         temp1.sort()
    #         if temp1 not in res:
    #             temp.append(temp1)
    #     res.extend(temp)
    # return res

def clumsy(N: int) -> int: # 4.1
    # 1006. 笨阶乘 medium
    stack = [N]
    op = 0
    for i in range(N-1,0,-1):
        if op == 0:
            stack.append(stack.pop()*i)
        elif op == 1:
            stack.append(int(stack.pop()/float(i)))
        elif op == 2:
            stack.append(i)
        elif op == 3:
            stack.append(-i)
        op = (op + 1) % 4
    return sum(stack)

def search(nums, target: int) -> bool: # 4.7
    # 81. 搜索旋转排序数组 II medium
    if nums is None:return False
    n = len(nums)
    for i in range(n):
        if target == nums[i]:
            return True
    return False



def findMin(nums) -> int: # 4.8
    # 153. 寻找旋转排序数组中的最小值 medium 二分查找
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left+right)//2
        if nums[mid] < nums[right]:
            right = mid
        else:
            left = mid + 1
    return nums[left]


def findMin2(nums) -> int: # 4.9
    # 154. 寻找旋转排序数组中的最小值 II hard 主要是有了重复值
    left, right = 0, len(nums)-1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]

def largestNumber(nums): # 4.12 https://blog.csdn.net/weixin_42001089/article/details/84197336
    # 179. 最大数 medium

    nums_str =list(map(str,nums))

    nums_str = sorted(nums_str,key=cmp_to_key(lambda x,y:int(x+y)-int(y+x)),reverse=True)
    return ''.join(nums_str if nums_str[0]!='0' else '0')

def isUgly(n: int):
    if n <= 0: return False
    num = [2, 3, 5]
    for i in num:
        while n % i == 0:
            n //= i
    return n == 1
    # if n <= 0:return False
    # while n % 2 == 0:
    #     n //= 2
    # while n % 3 == 0:
    #     n //= 3
    # while n % 5 == 0:
    #     n //= 5
    # return n==1


# def nthUglyNumber(n: int):

def rob(nums): # 4.15
    # 213. 打家劫舍 II medium
    # 先把1做了
    if sum(nums)==0 or not nums:return 0
    n = len(nums)
    if n < 3: return max(nums)
    def my_rob(numss):
        n = len(numss)
        dp = [0] * n
        dp[0],dp[1] = numss[0],numss[1]
        for i in range(2,n):
            dp[i] = max(dp[:(i-1)])+numss[i]
        return max(dp)
    return max(my_rob(nums[0:n-1]),my_rob(nums[1:n]))


def rob1(nums):  # 4.15
    # 198.打家劫舍 medium 和上一个的区别就是首尾未相连
    # if sum(nums) == 0:return 0
    # if len(nums)<3:return max(nums)
    #
    # n = len(nums)
    # dp = [0] * n
    # dp[0],dp[1] = nums[0],nums[1]
    # # dp[0],dp[1] = nums[0],max(dp[0],dp[1])
    # for i in range(2,n):
    #     dp[i] = max(dp[:(i-1)])+nums[i]
    # return max(dp)


    # `if not nums:  官方给的题解 是这个房屋偷与不偷   我上面自己写的 每个都偷 最后找出最大值
    #     return 0
    #
    # size = len(nums)
    # if size == 1:
    #     return nums[0]
    #
    # dp = [0] * size
    # dp[0] = nums[0]
    # dp[1] = max(nums[0], nums[1])
    # for i in range(2, size):
    #     dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
    #
    # return dp[size - 1]`

    # 滚动数组这个 也是偷与不偷
    if not nums:
        return 0
    n = len(nums)
    if n<3:return max(nums)
    first = nums[0]
    second = max(nums[0],nums[1])
    for i in range(2,n):
        first,second = second,max(first+nums[i],second)
    return second

@functools.lru_cache(None)
def isScramble(s1: str, s2: str):
    n = len(s1)
    if n==0:return True
    if n==1:return s1==s2
    if sorted(s1) != sorted(s2):
        return False
    for i in range(1,n):
        if isScramble(s1[:i],s2[:i]) and isScramble(s1[i:n],s2[i:n]):
            return True
        elif isScramble(s1[:i],s2[-i:]) and isScramble(s1[-i:],s2[:i]):
            return True
    return False

def removeElement(nums, val: int): # 4.19
    # 27. 移除元素 easy    竟然没做出来 双指针
    n = len(nums)
    first = 0
    # for i in range(n+1):
    #     if nums[i] == val:
    #         nums[i] = nums[]
    # while first <= last:
    #     while nums[last] == val:
    #         last -= 1
    #     while nums[first] != val:
    #         first += 1
    #     nums[first], nums[last] = nums[last], nums[first]
    #     first += 1
    #     last -= 1
    for i in range(n):
        if nums[i] != val:
            nums[first] = nums[i]
            first += 1
    return first

def strStr(haystack: str, needle: str):# 4.20
    # 28.实现strStr() easy
    ''' 这里没必要
    if len(needle) == 0:return 0
    if len(needle) == 1:
        for i in range(len(haystack)):
            if haystack[i] == needle:
                return i
        return -1'''

    # for j in range(len(haystack)-len(needle)+1): # 注意这里
    #     fisrt = True
    #
    #     for i in range(len(needle)):
    #         if haystack[j+i] != needle[i]:
    #             fisrt = False
    #             break
    #     if fisrt:
    #         return j
    # return -1
    m = len(haystack)
    n = len(needle)
    if n == 0: return n
    for i in range(m - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1

def numDecodings(s: str):
    if s == '0':return 0
    n = len(s)
    # if n == 1:return 1
    # a = 0
    # for i in range(2,n):
    #     if s[i] not in s[:i]:
    #         a += 1
    #     if s[i-2:i] not in s[:i-2] and int(s[i-2:i]) <= 26:
    #         a += 1
    # return a
    """ dp
    dp = [1] + [0]*n
    for i in range(1,n+1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        if i>1 and s[i-2]!='0' and int(s[i-2:i]) <= 26:
            dp[i] += dp[i-2]
    return dp[n]"""
    a,b,c = 0,1,0
    for i in range(1,len(s)+1):
        c = 0
        if s[i-1] != '0':
            c += b
        if i > 1 and s[i - 2]!= '0' and int(s[i - 2:i]) <= 26:
            c += a
        a,b = b,c
    return c

def maxSumSubmatrix(matrix, k: int):
    ans = float("-inf")
    m, n = len(matrix), len(matrix[0])

    for i in range(m):  # 枚举上边界
        total = [0] * n
        for j in range(i, m):  # 枚举下边界
            for c in range(n):
                total[c] += matrix[j][c]
    return total


def judgeSquareSum(c: int): # 4.28
    # 633. 平方数之和  medium 双指针
    left = 0
    right = int((math.sqrt(c)))
    while left<=right:
        s = left*left + right*right
        if s == c:
            return True
        elif s > c:
            right -= 1
        else:
            left += 1
    
    return False

def canCross(stones): # 4.29   失败
    # 403. 青蛙过河 hard
    n = len(stones)
    if n==2:return True
    for i in range(1,n):
        pre = stones[i] - stones[i-1]
        back = stones[i+1] - stones[i]
        if pre == back or pre == back-1 or pre == back+1:
            continue
        else:
            return False
    return True

def singleNumber(nums): # 4.30
    # 137. 只出现一次的数字 II

    res = list(set(nums))
    s = sum([i*3 for i in res])
    for i in res:
        if s - sum(nums) == 2*i:
            return i

def xorOperation(n: int, start: int): # 5.7
    # 1486.数组异或操作 easy
    # res = [start] * n
    # a = 0
    # for i in range(n):
    #     res[i] += 2*i
    # a = start
    # for i in range(1,n):
    #     a = a ^ res[i]
    # return a
    res = start
    for i in range(1,n):
        res = res ^ (start+2*i)
    return res


def minimumTimeRequired(jobs, k: int): # 5.8
    # 1723. 完成所有工作的最短时间 hard
    n = len(jobs)
    if k == n: return max(jobs)
    
def canEat(candiesCount, queries): # 6.1 我回来了
    # 1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？ medium
    total = list(accumulate(candiesCount))

    ans = list()
    for favoriteType, favoriteDay, dailyCap in queries:
        x1 = favoriteDay + 1
        y1 = (favoriteDay + 1) * dailyCap
        x2 = 1 if favoriteType == 0 else total[favoriteType - 1] + 1
        y2 = total[favoriteType]

        ans.append(not (x1 > y2 or y1 < x2))

    return ans

def isNumber(s: str) -> bool: # 617
    # 65. 有效数字  hard
    ab = ['+','-']
    if s[0] in ab and s[1] in ab:return False
    e = ['e','E']
    if s[0] in e:return False

    n = len(s)
    pos = -1
    pos1 = -1
    for i in range(n):
        if s[i] == '.':
            pos = i
        elif s[i] in e:
            pos1 = i
    if pos1 < pos:return False

def hammingWeight(n: int) -> int: # 6.23
    # 剑指 Offer 15. 二进制中1的个数 easy
    ans = 0
    for i in bin(n):
        if i == '1':
            ans += 1
    return ans

def majorityElement( nums) -> int: # 7.9
    # 面试题 17.10. 主要元素 easy    摩尔投票法
    if len(nums) == 1:return nums[0]
    for i in range(1,len(nums)):
        if i % 2 == 1:
            if nums[i-1] != nums[i]:
                nums[i-1],nums[i] = -1,-1
        else:
            continue
    for i in nums:
        if i != -1:
            return i
    return -1

def maximumElementAfterDecrementingAndRearranging(arr) -> int: # 715
    # 1846. 减小和重新排列数组后的最大元素 medium
    arr = sorted(arr)
    if arr[0] != 1:
        arr[0] = 1
    for i in range(1,len(arr)):
        if arr[i] - arr[i-1] > 1:
            arr[i] = arr[i-1] + 1
    return arr[-1]

def maxArea(height) -> int: #715
    # 11. 盛最多水的容器 medium
    left,right = 0,len(height)-1
    res = 0
    while left < right:
        area = min(height[left],height[right]) * (right-left)
        res = max(res,area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return res

def canJump( nums) -> bool: # 715 
    # 跳台阶 贪心
    # maxi = 0
    # for i, jump in enumerate(nums):
    #     if maxi>=i:
    #         maxi = max(i + jump,maxi)
    #     else:return False
    # return maxi >= i
    max_i = 0       #初始化当前能到达最远的位置
    for i, jump in enumerate(nums):   #i为当前位置，jump是当前位置的跳数
        if max_i>=i and i+jump>max_i:  #如果当前位置能到达，并且当前位置+跳数>最远位置  
            max_i = i+jump  #更新最远能到达位置
    return max_i>=i

def longestPalindrome(s: str) -> int: #716
    # 409. 最长回文串 easy   贪心
    # chr_freq = collections.Counter(s)

    # flag = 0        #最多允许一个出现频率freq为奇数的字母，置于中间
    # odd_freq_cnt = 0    #出现频率freq为奇数的字母个数
    # for chr, freq in chr_freq.items():
    #     if freq % 2 == 1:
    #         flag = 1
    #         odd_freq_cnt += 1
    # result = len(s) - odd_freq_cnt + flag
    
    # return result

    # 暴力
    # count = Counter(s)
    # jishu = 0
    # l = 0
    # for i in s.values():
    #     if i % 2 == 0:
    #         l += i
    #     else:
    #         if jishu == 0:
    #             l += i
    #             jishu = 1
    #         else:
    #             l += i-1
    # return l
    count = Counter(s)
    flag = 0
    freq = 0
    for i in count.values():
        if i % 2 == 1:
            flag = 1
            freq += 1
    return len(s) - freq + flag

def validPalindrome(s: str) -> bool: # 716
    # 680. 验证回文字符串 Ⅱ eady 贪心+双指针 
    # （暴力的话是逐个删掉字母，检查剩余的字符是否为回文串）
    left, right = 0, len(s)-1
    def check(low, high):
        l,r = low, high
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True
    while left < right:
        if s[left] == s[right]:
            left += 1
            right -= 1
        else:
            return check(left+1,right) or check(left,right-1)
    return True

def isPalindrome(s: str) -> bool: #716
    s = [i.lower() for i in s if i.isalnum()]
    # s = "".join(ch.lower() for ch in s if ch.isalnum())
    print('s:',s)
    left, right = 0, len(s)-1
    while left < right:
        if s[left] == s[right]:
            left += 1
            right -= 1
        else:
            return False

    return True

def minPairSum(nums) -> int: # 720
    # 1877. 数组中最大数对和的最小值 medium
    nums = sorted(nums)
    ans = 0
    i, j = 0, len(nums)-1
    while i < j:
        ans = max(ans,nums[i] + nums[j])
        i += 1
        j -= 1
    return ans


def isCovered(ranges, left: int, right: int) -> bool: # 723
    n = len(ranges)
    # if left < ranges[0][0] or right > ranges[n - 1][1]: return False
    res = [i for i in range(left, right + 1)]
    theshy = []
    for i in ranges:
        theshy.extend([z for z in range(i[0], i[1] + 1)])
    theshy = sorted(list(set(theshy)))
    for i in res:
        if i not in theshy:
            return False
        else:
            continue
    return True
    # mi = min([i[0] for i in ranges])
    # ma = max([i[1] for i in ranges])
    # c= 0
    # return left >= mi and right <= ma


def maximumTime(time: str) -> str: # 727
    # 1736. 替换隐藏数字得到的最晚时间 easy

    # hour = time.split(':')[0]
    # minute = time.split(':')[1]
    # print(hour)
    # if hour[0] == '?':
    #     hour[0] = '2'
    time = list(time)
    if time[0] == '?':
        time[0] = '1' if '4'<=time[1]<='9' else '2'
    if time[1] == '?':
        time[1] = '3' if time[0] == '2' else '9'
    if time[3] == '?':
        time[3] = '5'
    if time[4] == '?':
        time[4] = '9'
    a = ''
    for i in time:
        a += i
    return a

def restoreArray(adjacentPairs): # 727
    # 1743. 从相邻元素对还原数组 medium
    n = len(adjacentPairs)
    if n==1:return adjacentPairs[0]
    res = [adjacentPairs[0][0],adjacentPairs[0][1]]
    
    while len(res) != n+1:
        left = res[0]
        right = res[-1]
        for i in range(1,n):
            if adjacentPairs[i][0]==left:
                res.insert(0,adjacentPairs[i][1])
                left = adjacentPairs[i][1]
                continue
            if adjacentPairs[i][0] == right:
                res.append(adjacentPairs[i][1])
                right = adjacentPairs[i][1]
                continue
            if adjacentPairs[i][1] == left:
                res.insert(0,adjacentPairs[i][0])
                left = adjacentPairs[i][0]
                continue
            if adjacentPairs[i][1] == right:
                res.append(adjacentPairs[i][0])
                right = adjacentPairs[i][1]

    return res

def findUnsortedSubarray(nums) -> int: # 8.3
    # 581. 最短无序连续子数组 medium
    if len(nums) <= 1:return 0
    left, right = 0, len(nums)-1
    while left <= right:
        if nums[left] > nums[right]:
            return len(nums[left:right])
        else:
            left += 1
            right -= 1
    return 0

def triangleNumber(nums) -> int: # 8.4
    # 611. 有效三角形的个数 medium
    def istriangle(a,b,c):
        if a+b<c or a+c<b or b+c<a:
            return False
        else:
            if abs(a-b)>c or abs(a-c)>b or abs(b-c)>a:
                return False
        return True
    for i in range(len(nums)):
        pass
    return 

def nthUglyNumber(n: int) -> int:
    num = [2,3,5]
    i = 0
    j = 0
    res = [1]
    while j<=n:
        pass
        

    return 1

def numberOfArithmeticSlices(nums) -> int: # 8.10
    # 413. 等差数列划分 medium    # 注意是连续的 前面理解错了
    # n = len(nums)
    # if n == 1:return 0
    # dp = [0] * n
    # for i in range(2,n):
    #     if nums[i-2]-nums[i-1] == nums[i-1]-nums[i]:
    #         dp[i] = dp[i-1]+1
    #     else:
    #         pass
    # return sum(dp)
    n = len(nums)
    if n == 1:return 0
    ans = 0
    a = 0
    for i in range(2,n):
        if nums[i-2]-nums[i-1] == nums[i-1]-nums[i]:
            a += 1
        else:
            a = 0
        ans += a 
    return ans

def numberOfArithmeticSlices2(nums) -> int: # 8.11
    # 413. 等差数列划分er hard
    ans = 0
    f = [defaultdict(int) for _ in nums]
    for i, x in enumerate(nums):
        for j in range(i):
            d = x - nums[j]
            cnt = f[j][d]
            ans += cnt
            f[i][d] += cnt + 1
    return ans

def longestPalindromeSubseq(s: str): # 8.12
    # 516. 最长回文子序列 medium 经典dp
    n = len(s)
    if n==1:return 1
    dp = [[0]*n for _ in range(n)]
    for i in range(n-1,-1,-1):
        dp[i][i] = 1
        for j in range(i+1,n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j],dp[i][j-1])
    return dp[0][n-1]

# def rob(nums) -> int: # 8.16
#     # 198. 打家劫舍 medium
    

def checkRecord(s: str) -> bool: 
    if s.count('A') >= 2:return False
    if len(s)<=2:return True
    else:
        for i in range(len(s)-2):
            if s[i] == 'L' and s[i+1] == 'L' and s[i+2] == 'L':
                return False
    return True

def reverseVowels(s: str) -> str: # 819
    # 345. 反转字符串中的元音字母 easy
    # s = list(s)
    # res = ['a','e','i','o','u','A','E','U','I','O']
    # c = []
    # for i in s:
    #     if i in res:
    #         c.append(i)
    # c = c[::-1]
    # z = ''
    # for i in range(len(s)):
    #     if s[i] in res:
    #         z += c[0]
    #         c.remove(c[0])
    #     else:
    #         z += s[i]
    # return z
    def ziji(ch):
        return ch in ['a','e','i','o','u','A','E','U','I','O']
    
    n = len(s)
    left, right =  0, n-1
    s = list(s)
    while left < right:
        while left<n and not ziji(s[left]):
            left += 1
        while right>0 and not ziji(s[right]):
            right -= 1
        if left < right:
            s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return ''.join(s)
    
def reverseStr(s: str, k: int) -> str:
    # def rever(nums,k):
    #     nums_k = nums[:k]
    #     nums_k = nums_k[::-1]
    #     nums[:k] = nums_k
    #     return nums
    # s = list(s)
    # n = len(s)
    # res = []
    # result = []
    # for i in range(n):
    #     if i%(2*k) == 0:
    #         res = rever(res,k)
    #         result.extend(res)
    #         res = []
    #     else:
    #         res.append(s[i])
    # def rever(nums,k):
    #     nums_k = nums[:k]
    #     nums_k = nums_k[::-1]
    #     nums[:k] = nums_k
    #     return nums

    # return result
    s = list(s)
    for i in range(0,len(s),2*k):
        s[i:i+k] = reversed(s[i:i+k])
    return ''.join(s)


def numRescueBoats(people, limit: int) -> int:
    people.sort()
    n = len(people)
    res = 0
    left, right = 0, n-1
    while left < right:
        if people[left] + people[right] > limit:
            right -= 1
        else:
            right -= 1
            left += 1
        res += 1

    return res



def ceshi():
    # min_depth = 10**9
    # print('min_depth',min_depth) 
    # que = collections.deque([(2, 1)])
    # print(que)
    pass





if __name__ == '__main__':
    # obj = reverse(123)
    # obj = myPow(2.00000,10)
    # obj = lengthOfLongestSubstring('abcabcbb')
    # obj = countBits(5)

#------------------------------------------------------------
# 动态规划
    #obj = climbStairs(5)
    # obj = uniquePaths(3,7)
    # obj = maxEnvelopes([[5,4],[6,5],[6,7],[2,3]])
    # obj = lengthOfLIS([10,9,2,5,3,7,101,18])
    # obj = minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]])
    # obj = minPathSum([[1]])
    # obj = longestPalindrome('babad')
    # obj = numDistinct(s = "babgbag", t = "bag")
    # obj = longestPalindrome('babad')
#------------------------------------------------------------
# 单调栈问题
    # obj = nextGreaterElements([1,2,1])
    # obj = nextGreaterElement([4,1,2],[1,3,4,2])


#------------------------------------------------------------
# 贪心 (双指针)
    # obj = maxArea([1,1])
    # obj = canJump([2,3,1,1,4])
    # obj = longestPalindrome("abccccdd")
    # obj = validPalindrome('abbbbbacd')
    # obj = isPalindrome('0p')
    # print(obj)




#------------------------------------------------------------
    # obj = isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#")
    # obj = removeDuplicates('abbaca')
    # obj = spiralOrder([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    # obj = generateMatrix(3)
    # obj = hammingWeight(00000000000000000000000000001011)
    # obj = find132pattern([-1, 3, 2, 0])
    # obj = subsetsWithDup([1,2,3])
    # obj = clumsy(4)
    # obj = search([1,5,9,8,7,2],2)
    # obj = findMin([4,5,6,7,0,1,2])
    # obj = findMin2([1,2,2,2])
    # obj = largestNumber([10,2])
    # obj = isUgly(6)
    # obj = rob1([1,2,3,1])
    # obj = rob([1,2,3,1])
    # obj = isScramble("abc","bca")
    # obj = removeElement([0,1,2,2,3,0,4,2],2)
    # obj = strStr('hello','ll')
    # obj = numDecodings('122')
    # obj = maxSumSubmatrix([[1,0,1],[0,-2,3]], k = 2)
    # obj = judgeSquareSum(10)
    # obj = canCross(stones = [0,1,3,5,6,8,12,17])
    # obj = singleNumber([2,2,3,2])
    # obj = xorOperation(10,5)
    # obj = minimumTimeRequired([3,2,3], 3)
    # obj = isNumber('--6')
    # obj = hammingWeight()
    # obj = majorityElement([8,8,7,7,7])
    # obj = maximumElementAfterDecrementingAndRearranging([100,1,1000])

    # obj = minPairSum([3,5,2,3])
    # obj = isCovered([[25,42],[7,14],[2,32],[25,28],[39,49],[1,50],[29,45],[18,47]], left = 15, right = 38)
    # obj = maximumTime("0?:3?")
    # obj = restoreArray([[-3,-9],[-5,3],[2,-9],[6,-3],[6,1],[5,3],[8,5],[-5,1],[7,2]])
    # obj = findUnsortedSubarray([2,6,4,8,10,9,15])
    # obj = numberOfArithmeticSlices([1,2,3,5,7,8,9])
    # obj = numberOfArithmeticSlices2([1,2,3,5,7,8,9])
    # obj  = longestPalindromeSubseq('abbbb')
    # obj = rob([1,2,3,1,5,8,9])
    # obj = checkRecord("AALL")
    # obj = reverseVowels('hello')
    # obj = reverseStr('abcdefg',2)
    obj = numRescueBoats(people=[3,5,3,4], limit=5)


    print(obj)

    # print(max(3,3))



    ceshi()