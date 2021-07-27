import numpy as np




class Solution:
    def findKthLargest(self, nums, k: int) -> int:
        # 215. 数组中的第K个最大元素 medium



        return 3

    def permute(self, nums): # 4.17
        # 46. 全排列
        '''方向错了 自己想的是动态规划 其实使用递归回溯'''
        # n = len(nums)
        # if n == 1:return [nums]
        # col = 1
        # for i in range(1,n+1):
        #     col *= i
        # dp = [[0]*n for _ in range(col)]
        def backtrack(first=0):
            # 所有数都填完了
            if first == n:
                res.append(nums[:])
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]

        n = len(nums)
        res = []
        backtrack()
        return res

    def searchMatrix(self, matrix, target: int) -> bool: # 4.19
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[0])):
        #         if matrix[i][j] == target:
        #             return True
        # return False

        ''' 这个方法绝了
        l = len(matrix[0])
        w = len(matrix)
        col = 0
        row = w - 1
        while col < l and row >= 0:
            if matrix[row][col]>target:
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                return True
        return False

        '''





if __name__ == '__main__':
    # obj = Solution()
    # # result = obj.findKthLargest()
    # # result = obj.permute([1,2,3])
    # result = obj.searchMatrix([[1,4,7,11,15],[2,5,8,12,19],
    #                            [3,6,9,16,22],[10,13,14,17,24],
    #                            [18,21,23,26,30]], target=11)
    #
    #
    # print(result)
    a = np.random.default_rng()
    print(a)
