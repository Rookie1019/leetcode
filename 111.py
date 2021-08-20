import pandas as pd
df = pd.DataFrame([  
            ['green' , 'A'],   
            ['red'   , 'B'],   
            ['blue'  , 'A']])  

df.columns = ['color',  'class'] 
pd.get_dummies(df) 
# ————————————————
# 版权声明：本文为CSDN博主「魔术师_」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/maymay_/article/details/80198468def merge_sort(array):
#     if len(array) == 1:
#         return array
#     left_array = merge_sort(array[:len(array)//2])
#     right_array = merge_sort(array[len(array)//2:])
#     return merge(left_array, right_array)
 
 
def merge(left_array, right_array):
    left_index, right_index, merge_array = 0, 0, list()
    while left_index < len(left_array) and right_index < len(right_array):
        if left_array[left_index] <= right_array[right_index]:
            merge_array.append(left_array[left_index])
            left_index += 1
        else:
            merge_array.append(right_array[right_index])
            right_index += 1
    merge_array = merge_array + left_array[left_index:] + right_array[right_index:]
    return merge_array
 
 
if __name__ == '__main__':
    array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
    # print(merge_sort(array))
    a=eval(input()) 
    print(a)
