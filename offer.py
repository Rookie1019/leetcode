
def findNumberIn2DArray(matrix, target: int):
    # 剑指 Offer 04. 二维数组中的查找
    if matrix is None or len(matrix)==0 or len(matrix[0])==0:return False
    n = len(matrix)-1
    m = 0
    founded = False
    
    while n>-1 and not founded:
        if target == matrix[n][m]:
            founded = True
        else:
            if target > matrix[n][m]:
                m += 1
                if m >= len(matrix[0]):
                    break
            if target < matrix[n][m]:
                n -= 1
    return founded

def numWays(n: int) -> int:
        if n == 0 or n == 1:return 1
        if n == 2:return 2

        res = [0] * (n+1)
        res[1] = 1
        res[2] = 2

        for i in range(3,n+1):
            res[i] = res[i-1] + res[i-2]
        return res[-1] % 1000000007

def exist(board, word: str) -> bool:
    # 剑指 Offer 12. 矩阵中的路径 medium
    


if __name__ =='__main__':
    
    # obj = findNumberIn2DArray(matrix=[
    #                             [1,   4,  7, 11, 15],
    #                             [2,   5,  8, 12, 19],
    #                             [3,   6,  9, 16, 22],
    #                             [10, 13, 14, 17, 24],
    #                             [18, 21, 23, 26, 30]
    #                             ],target=100)
    obj = numWays(44)
                                    
    print(obj)
