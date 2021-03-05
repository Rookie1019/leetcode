import math



class Search_muti():
    def __init__(self,a:int,L:list):
        self.L = L
        self.a = a


    def sequential_searnch(self):
        # 无序表搜索
        d = False
        for i in self.L:
            if i == self.a:
                d = True
                break
        return d

    def ordered_search(self):
        # 有序表搜索
        found = False
        for j,i in enumerate(self.L):
            if i > self.a:
                return found
            else:
                if i == self.a:
                    found = True
                    break
            print(j)
        return found

    def Binary_search(self,x:int,y:list):
        # pos = len(y) // 2
        # if x < y[pos]:
        #     return self.Binary_search(x,y[:pos])
        # elif y[pos] == x:
        #     return True
        # else:
        #     return self.Binary_search(x,y[pos+1:])
        # if y[pos] == x:
        #     return True
        # else:
        #     if y[pos] < x:
        #         return self.Binary_search(x,y[pos+1:])
        #     else:
        #         return self.Binary_search(x,y[:pos])
        first = 0
        last = len(y)-1
        found = False

        while first <= last and not found:
            midpoint = (first + last)//2
            if y[midpoint] == x:
                found = True
            else:
                if x < y[midpoint]:
                    last = midpoint - 1
                else:
                    first = midpoint + 1
        return found


class sort_ziji():
    def __init__(self,nums:list):
        self.nums = nums

    def bubbleSort(self):
        ''''
        冒泡排序
        '''

        for j in range(len(self.nums)-1,0,-1):
            for i in range(j):
                if self.nums[i] < self.nums[i+1]:
                    continue
                else:
                    self.nums[i],self.nums[i+1] = self.nums[i+1],self.nums[i]
                # self.nums[i] = min(self.nums[i],self.nums[i+1])

        return self.nums


    def select_sort(self):
        '''
        选择排序
        '''
        for i in range(len(self.nums)-1,0,-1):
            pos = 0
            for j in range(i+1):
                if self.nums[j] > pos:
                    pos = self.nums[j]
                    z = j
            self.nums[z] = self.nums[i]
            self.nums[i] = pos

        return self.nums

        # 书上的写法
        # for fillslot in range(len(self.nums) - 1, 0, -1):
        #     positionOfMax = 0
        #     for location in range(1, fillslot + 1):
        #
        #         if self.nums[location] > self.nums[positionOfMax]:
        #             positionOfMax = location
        #         temp = self.nums[fillslot]
        #         self.nums[fillslot] = self.nums[positionOfMax]
        #         self.nums[positionOfMax] = temp
        # return self.nums

    def insert_sort(self):
        '''
        插入排序
        '''
        # new_nums = [self.nums[0]]
        # for i in range(len(self.nums)-1,0,-1):
        #     for j in new_nums:

        n = len(self.nums)
        # 从右边的无序序列中取出多少个元素执行这样的过程
        for j in range(1, n):
            # j = [1, 2, 3, n-1]
            # i 代表内层循环起始值
            i = j
            # 执行从右边的无序序列中取出第一个元素，即i位置的元素，然后将其插入到前面的正确位置中
            while i > 0:
                if self.nums[i] < self.nums[i-1]:
                    self.nums[i], self.nums[i-1] = self.nums[i-1], self.nums[i]
                    i -= 1
                else:
                    break
        return self.nums

    def quick_sort(self,first,last):
        mid_value = self.nums[first]

        n = len(self.nums)
        low = first
        high = last
        
        while low < high:
            while low < high and self.nums[high] > mid_value:
                high -= 1
            self.nums[low] = self.nums[high]
            low += 1

            while low < high and self.nums[low] < mid_value:
                low += 1
            self.nums[high] = self.nums[low]
            high -= 1
        

        # 循环结束时
        self.nums[low] = mid_value

        # 对low左边执行快排
        self.quick_sort(self.nums,first,low-1)

        # 对low的右边快排
        self.quick_sort(self.nums,low+1,last)
        
#################################################################
# 树

class Node(object):
    '''
    节点类
    '''
    def __init__(self,item):
        self.elem = item
        self.lchild = None #左节点
        self.rchild = None #右节点

class tree(object):
    '''
    二叉树
    '''
    def __init__(self):
        self.root = None
    
    # 按照广度优先的原则添加元素
    def add(self, item):
        node = Node(item)
        queue = [self.root]
        if self.root is None:  # 判断是不是空树 空的话直接加在根节点
            self.root = node
            return  
        while queue:
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return
            else:
                queue.append(cur_node.lchild)
            
            if cur_node.rchild is None:
                cur_node.rchild = node
                return
            else:
                queue.append(cur_node.rchild)

    def breadth_travel(self):
        '''广度优先遍历'''
        if self.root is None:
            return
        queue = [self.root]
 
        while queue:
            cur_node = queue.pop(0)
            print(cur_node.elem)
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def preorder(self, node):
        '''
        深度优先遍历
            1. 前序    根左右   preorder
            2. 中序    左根右
            3. 后序    左右根
        '''
        if node is None:
            return
        print(node.elem)
        self.preorder(node.lchild)
        self.preorder(node.rchild)
    
    def inorder(self, node):
        '''中序 : 左根右'''
        if node is None:
            return
        self.inorder(node.lchild)
        print(node.elem)
        self.inorder(node.rchild)
    
    def backorder(self, node):
        '''后序 : 左右根'''
        if node is None:
            return
        self.backorder(node.lchild)
        self.backorder(node.rchild)
        print(node.elem)


###########################################################
# 链表

class list_Node(object):
    '''链表节点类'''
    def __init__(self,elem):
        self.elem = elem
        self.next = None

class SingleLink(object):
    '''单链表'''
    def __init__(self, node=None):
        self._head = node

    def is_empty(self):
        '''链表是否为空'''
        return self._head == None

    def length(self):
        '''返回链表长度'''
        count = 0 # 计数
        cur = self._head # 游标指针
        while cur != None:
            count += 1
            cur = cur.next # .next就指的是下一个区域
        return count

    def travel(self):
        '''遍历整个链表'''
        cur = self._head
        while cur != None:
            print(cur.elem,end=' ')
            cur = cur.next

    def add(self, item):
        '''链表头部添加元素'''
        node = list_Node(item)
        node.next = self._head
        self._head = node   

    def append(self, item):
        '''链表尾部添加元素'''
        node = list_Node(item)
        if self.is_empty():  # 判断是否为空
            self._head = node
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node


    def insert(self, pos, item):
        '''向指定位置添加元素'''

        if pos <= 0: # 小于等于0的时候 默认加在头部
            self.add(item=item)  
        elif pos >= self.length():
            self.append(item=item)
        else:
            node = list_Node(item)
            pre = self._head # 前一个结点
            for i in range(pos-1):
                pre = pre.next
            node.next = pre.next
            pre.next = node
        

    def remove(self, item):
        '''删除节点'''

    def search(self, item):
        '''查找结点是否存在'''
        if self.length() == 0:
            return False
        else:
            cur = self._head
            while cur != None:
                if cur.elem == item:
                    return True
                else:
                    cur = cur.next
            return False


if __name__ == '__main__':
    # x = Search_muti(a=10,L=[1,3,4,45,67,98,109])
    # v = x.sequential_searnch()
    # v = x.ordered_search()
    # v = x.Binary_search(10,[1,3,4,45,67,98,109])
    # print(v)

    # 排序类
    a = sort_ziji([1,8,7,6,5,9,4])
    # c = a.bubbleSort()
    # c = a.select_sort()
    # c = a.insert_sort()
    a.quick_sort(0,6)
    print('排好序：',a)
    # print(c)


    # tree = tree()
    # tree.add(3)
    # tree.add(56)
    # tree.add(72)
    # tree.add(35)
    # tree.add(24)
    # tree.add(67)
    # tree.add(45)
    # tree.add(6)
    # tree.add(2)
    # tree.add(7)
    # tree.add(56)
    # tree.add(78)
    # tree.add(72)
    # tree.add(69)
    # tree.add(90)
    # tree.breadth_travel()
    
    # a = list_Node(3)

    ll = SingleLink()
    
    a = ll.is_empty()
    print(a)
    ll.append(4)
    ll.append(5)
    ll.add(3)
    ll.add(8)
    ll.append(9)
    ll.append(19)
    ll.insert(3,25)
    ll.insert(100,24)
    ll.insert(-9,100)

    a = ll.search(100)
    print(a)

    ll.travel()

    
    



