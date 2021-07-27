
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
            


    return 1



if __name__ == '__main__':
################################################################
    # 动态规划
    # obj = canJump([3,2,1,0,4])
    
    
################################################################
    # 贪心算法
    # obj = canJump([3,2,1,0,4])

################################################################
    # 数学技巧
    # obj = mySqrt(15)
################################################################
    # 滑动窗口
    obj = threeSumClosest([-1,2,1,-4],1)

################################################################
    # 哈希算法


################################################################
    # 二分法


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
