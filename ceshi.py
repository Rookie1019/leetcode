import re

# class a():
#     def __init__(self,item) -> None:
#         self.item = item
#         self.v = None
#         self.f = None
#
# a = a(3)
# print(a.v)


def erfen(nums,x):
    left = 0
    right = len(nums)-1
    found = False

    while left <= right and not found:
        mid = (left+right)//2
        if nums[mid] == x:
            found = True
        else:
            if x < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
    return found
def fo():
    for i in range(3):
        print(i)


if __name__ == '__main__':
    a = erfen([1,2,3,6,7],6)
    print(a)
    fo()









