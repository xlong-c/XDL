# 二叉树,根节点,123,根节点到叶子节点的所有数之和

from __future__ import annotations

class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: TreeNode | None = None,
        right: TreeNode | None = None,
    ) -> None:
        self.val = val
        self.left = left
        self.right = right
    
def sum_tree(root: TreeNode | None, current: int = 0) -> int:
    if root is None:
        return 0

    current = current * 10 + root.val
    if root.left is None and root.right is None:
        return current

    return sum_tree(root.left, current) + sum_tree(root.right, current)

if __name__ == "__main__":
    # 创建一个简单的二叉树
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4) #124
    root.left.right = TreeNode(5) #125
    root.right.left = TreeNode(6) #136
    root.right.right = TreeNode(7) #137

    print("Sum of all numbers from root to leaf nodes:", sum_tree(root))
    
    
