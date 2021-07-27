
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:# 3.25
        # [1,1,2,3,3] -> [2]
        if not head:return head

        dummy = ListNode(0, head)
        
        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                x = cur.next.val
                while cur.next and cur.next.val==x:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next

    def deleteDuplicates(self, head: ListNode) -> ListNode:# 3.26
        # [1,1,2,3,3] -> [1,2,3]
        if not head:return head

        cur = head
        while cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def deleteNode(self, head: ListNode, v: int) -> ListNode: # 3.26
        # 剑指 Offer 18. 删除链表的节点
        cur = head
        if head.val == v:return head.next
        while cur.next:
            if cur.next.val == v:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def rotateRight(self, head: ListNode, k: int) -> ListNode: # 4.1
        # 旋转链表 medium
        if head is None or head.next is None: return head
        cur = head
        num = 1
        while cur.next:
            cur = cur.next
            num += 1
        k = k % num
        for _ in range(k):
            pre = head.next
            back = head
            while pre.next:
                pre = pre.next
                back = back.next
            pre.next = head
            back.next = None
            head = pre
        return head

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:return None
        pa, pb = headA, headB
        while pa != pb:
            pa=headB if pa is None else pa.next
            pb=headA if pb is None else pb.next
        
        return pa    


class TreeNode:


    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution413:
    # 783.二叉搜索树节点最小距离
    def minDiffInBST(self, root: TreeNode) -> int:
        self.vals = []
        self.inorder(root)
        a = self.vals[1] - self.vals[0]
        for i in range(2, len(self.vals)):
            a = min(self.vals[i] - self.vals[i - 1], a)
        return a

    def inorder(self, node: TreeNode):
        if node is None:
            return
        if node.left is not None:
            self.inorder(node.left)
        self.vals.append(node.val)
        if node.right is not None:
            self.inorder(node.right)

if __name__ == '__main__':
    # a = ListNode([1,2,3,3,4,4,5])
    # s = Solution()
    # obj = s.deleteDuplicates(a)
    a,b = ListNode([4,1,8,4,5]),ListNode([5,0,1,8,4,5])
    s = Solution()
    obj = s.getIntersectionNode(a,b)

    print(obj)












