c = lambda x : x**2
# print(c(6))

def aaa(s):
    a = s.split(' ')
    # print(type(s))
    # a = []
    # for i in s:
    #     a.append(a)
    print(a)
    d = range(5)
    print('range',d)

# aaa('dasda dsaffd gfgf,dsad')

def pre(nums):
    print(nums)
    preSum = [0]
    for num in nums:
        preSum.append(preSum[-1] + num)
    print(preSum)
    print(preSum[3]-preSum[0]) # 0,2
    print(preSum[6]-preSum[2]) # 2,5

    print(preSum[6]-preSum[0]) # 0,5
    
def preSum_mat(matrix):
    # print(len(matrix))
    new = [[0] for _ in range(len(matrix))]
    # for i in range(len(matrix)):
    #     new.append([0])
    print(new)
    
    for i in range(len(matrix)):
        # new.append([0])
        for j in matrix[i]:
            new[i].append(new[i][-1]+j)
            

    print('new',new)
    D = [ [0] * (5 + 1) for _ in range(7 + 1)]
    
    print('D',D)

# pre([-2, 0, 3, -5, 2, -1])
preSum_mat([
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
])

