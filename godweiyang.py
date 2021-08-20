
## #https://mp.weixin.qq.com/s/MwCcV40zBpcfQ_LB39Z2ew

from collections import Counter
import math


################################################################
# 动态规划
def canJump(nums) -> bool: # 55
    # l = len(nums)
    # dp = [0] * l
    # dp[0] = 1
    # for i in range(l):
    #     if not dp[i]:
    #         return False
    #     if i+nums[i] >= l-1:
    #         return True
    #     for j in range(i+1,nums[i]+i):
    #         dp[j] = 1
    # return False
    maxi = 0
    for i, jump in enumerate(nums):
        if maxi>=i:
            maxi = max(i + jump,maxi)
        else:return False
    return maxi >= i

################################################################
# 数学方法
def mySqrt(x: int) -> int:
    # if x == 0:return 0
    # if x == 1:return 1
    # z = int(x/2)
    # for i in range(z):
    #     if i**2 == x:
    #         return i
    #     if i**2 < x and (i+1)**2 > x:
    #         return i
    
    # 数学方法 
    # if x == 0:return 0
    # ans = int(math.exp(0.5*math.log(x)))
    # return ans+1 if (ans+1)**2<=x else ans

    # 二分法
    l,r,ans = 0,x,-1
    while l<=r:
        mid = (l+r)//2
        if mid*mid <= x:
            ans = mid
        else:
            if mid**2 > x:
                r = mid-1               
            else:
                l = mid+1
                ans = mid
    return ans

################################################################
# 滑动窗口
def threeSumClosest( nums, target: int) -> int:
        l = len(nums)
        if l == 3:return sum(nums)
        # for i in range(l):
        #     res = abs(target-nums[i])
        #     b = nums.copy()
        #     b.remove(nums[i])
        nums = sorted(nums)
        ans = abs(target-(nums[0]+nums[1]+nums[2]))
        for i in range(3,l):
            c = abs(target-(nums[i-2]+nums[i-1]+nums[i]))
            ans = min(c,ans)
        return c

################################################################
# 哈希算法
def firstMissingPositive(nums) -> int:
# 41. 缺失的第一个正数  hard
    n = len(nums)
    if 1 not in nums:
        return 1
    
    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = 1
    
    for i in range(n):
        res = abs(nums[i])-1
        nums[res] = -abs(nums[res])
    
    for i in range(n):
        if nums[i] > 0:
            return i+1
    return n+1

################################################################
# 二分
def findMin(nums) -> int:
    left, right = 0, len(nums)-1
    while left < right:
        mid = left + (right-left)//2
        if nums[mid] < nums[right]:
            right = mid
        else:
            left = mid+1
    return nums[left]

################################################################
# 贪心
def maxProfit(prices) -> int:
    # n = len(prices)
    # if n == 0:return 0
    # min_input = prices[0]
    # max_profit = 0
    # for p in prices[1:]:
    #     min_input = min(p, min_input)
    #     max_profit = max(max_profit, p - min_input)

    # return max_profit
    n = len(prices)
    if n == 0:return 0 
    minn, res = prices[0], 0
    for i in range(1,n):
        res = max(res, prices[i]-minn)
        minn = min(minn, prices[i])
    return res

def maxProfit2(prices) -> int: # 8.6
    # 122. 买卖股票的最佳时机 II easy
    n = len(prices)
    if n==0:return 0
    res = 0
    for i in range(1,n):
        res += max(prices[i]-prices[i-1],0) # a b c (c-b)+(b-a)=c-a
    return res
            



if __name__ == '__main__':
################################################################
    ''' 我来解释下2个问题，1： 为啥状态方程这样对？ 2：怎么想到这样的状态方程？
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
        类似的题比如有 10：正则表达式匹配 44：通配符匹配 编辑距离 1143：最长公共子序列等等的 还有几道想不起来了'''
    # 动态规划
    # obj = canJump([3,2,1,0,4])
    
    
################################################################
    # 贪心算法
    # obj = canJump([3,2,1,0,4])
    # obj = maxProfit([7,6,4,3,1])
    obj = maxProfit2([5,4,3,2,1])

################################################################
    # 数学技巧
    # obj = mySqrt(15)
################################################################
    # 滑动窗口
    # obj = threeSumClosest([-1,2,1,-4],1)

################################################################
    # 哈希算法


################################################################
    # 二分法
    # obj = findMin([3,4,5,1,2])

################################################################
    # 单调栈（队列）


################################################################
    # dfs bfs


################################################################
    # 位运算

################################################################
    # 字符串

################################################################
    # 二叉树

################################################################
    # 并查集

################################################################
    # 数据结构

################################################################
    # 模拟

################################################################
    # 合集
    



    print(obj)

    print(100//2)
