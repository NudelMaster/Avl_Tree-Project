# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info


"""A class represnting a node in an AVL tree"""
import sys


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type key: int or None
    @param key: key of your node
    @type value: any
    @param value: data of your node
    """

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0

    """returns the key

    @rtype: int or None
    @returns: the key of self, None if the node is virtual
    """

    def get_key(self):
        return self.key if self.is_real_node() else None

    """returns the value

    @rtype: any
    @returns: the value of self, None if the node is virtual
    """

    def get_value(self):
        return self.value if self.key else None

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child (if self is virtual)
    """

    def get_left(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child (if self is virtual)
    """

    def get_right(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def get_parent(self):
        return self.parent

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def get_height(self):
        return self.height

    """returns the size of the subtree

    @rtype: int
    @returns: the size of the subtree of self, 0 if the node is virtual
    """

    def get_size(self):
        return self.size

    """sets key

    @type key: int or None
    @param key: key
    """

    def set_key(self, key):
        self.key = key

    """sets value

    @type value: any
    @param value: data
    """

    def set_value(self, value):
        self.value = value

    """sets left child

    @type node: AVLNode
    @param node: a node
    """

    def set_left(self, node):
        self.left = node

    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def set_right(self, node):
        self.right = node

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def set_parent(self, node):
        self.parent = node

    """sets the height of the node

    @type h: int
    @param h: the height
    """

    def set_height(self, h):
        self.height = h

    """sets the size of node

    @type s: int
    @param s: the size
    """

    def set_size(self, s):
        self.size = s

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def is_real_node(self):
        return self.height > -1


"""
A class implementing an AVL tree.
"""


class AVLTree(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        virtualNode = AVLNode(None, -1)
        self.root = None
        self.virtualNode = virtualNode

    # add your fields here

    """searches for a value in the dictionary corresponding to the key

    @type key: int
    @param key: a key to be searched
    @rtype: any
    @returns: the value corresponding to key.
    """

    def search(self, key):
        node = self.get_root()
        if node.is_real_node():
            def search_rec(node, key):
                if node.get_key() == key:
                    return node
                if node.left.is_real_node() and key > node.left.key():
                    return search_rec(node.left, key)
                if node.right.is_real_node() and key < node.right.key():
                    return search_rec(node.right, key)
        return None

    def updateHeight(self, node):
        left_height = max(node.get_left().get_height(), 0)
        right_height = max(node.get_right().get_height(), 0)
        if self.is_leaf(node):
            node.set_height(0)
        else:
            node.set_height(1 + max(left_height, right_height))

    def is_leaf(self, node):
        return node.get_left() == self.virtualNode and node.get_right() == self.virtualNode

    def updateSize(self, node):
        node.set_size(1 + node.left.size, node.right.size)

    """inserts val at position i in the dictionary

    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: any
    @param val: the value of the item
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, key, val):
        root = self.get_root()
        if root is None:
            self.root = AVLNode(key, val)
            self.root.set_left(self.virtualNode)
            self.root.set_right(self.virtualNode)
            self.updateHeight(self.root)
            return 0
        rb_num = 0

        def insert_rec(node, key, val, parent):  # regular insert to binary tree
            if key < node.get_key():
                if not node.get_left().is_real_node():
                    node.set_left(AVLNode(key, val))
                    node.get_left().set_parent(node)
                    return node.get_left()
                return insert_rec(node.get_left(), key, val, node)
            elif key > node.get_key():
                if not node.get_right().is_real_node():
                    node.set_right(AVLNode(key, val))
                    node.get_right().set_parent(node)
                    return node.get_right()
                return insert_rec(node.get_right(), key, val, node)

        y = insert_rec(root, key, val, self.virtualNode)  # get inserted node
        y.set_left(self.virtualNode)
        y.set_right(self.virtualNode)
        self.updateHeight(y)
        y = y.get_parent()
        while y:  # check if we the node is not null (happens after we get the parent of the root)
            temp_height = y.get_height()
            self.updateHeight(y)
            balance_factor = self.BFS(y)
            if abs(balance_factor) < 2 and y.get_height() == temp_height:
                break
            elif abs(balance_factor) < 2 and y.get_height() != temp_height:
                y = y.get_parent()
            elif abs(balance_factor) == 2:
                print("\n Before rotate")
                self.display(self.root)
                y, rb_num = self.rotate(y, balance_factor)
                y = y.get_parent()
                print("\n After rotate")
                self.display(self.root)
        return rb_num

    def left_rotate(self, node):
        B = node
        A = B.get_right()
        B.set_right(A.get_left())
        B.get_right().set_parent(B)
        A.set_left(B)
        self.fix_parent(A, B)
        self.updateHeight(B)
        self.updateHeight(A)
        return A

    def right_rotate(self, node):
        B = node
        A = B.get_left()
        B.set_left(A.get_right())
        B.get_left().set_parent(B)
        A.set_right(B)
        self.fix_parent(A, B)
        self.updateHeight(B)
        self.updateHeight(A)
        return A

    def fix_parent(self, A, B):
        A.set_parent(B.get_parent())
        if B == self.root:
            self.root = A
        elif B.get_parent().get_left().get_key() == B.get_key():
            A.get_parent().set_left(A)
        else:
            A.get_parent().set_right(A)
        B.set_parent(A)

    def BFS(self, node):
        return node.get_left().height - node.get_right().height

    def rotate(self, node, BFS):
        rb_count = 0
        balance_factor = BFS
        if balance_factor == 2:
            if self.BFS(node.get_left()) == -1:
                node = self.left_then_right(node.get_left())
                rb_count += 2
            else:
                node = self.right_rotate(node)
                rb_count += 1
        elif balance_factor == -2:
            if self.BFS(node.get_right()) == 1:
                node = self.right_then_left(node.get_right())
                rb_count += 2
            else:
                node = self.left_rotate(node)
                rb_count += 1
        return node, rb_count

        # Print the tree

    def display(self, root):
        lines, *_ = self._display_aux(root)
        for line in lines:
            print(line)

    def _display_aux(self, node):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if not node.get_right() and not node.get_left():
            if not node.is_real_node():
                line = '%s' % "V"
            else:
                line = '%s' % node.get_key()

            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if not node.get_right() and node.get_left():
            lines, n, p, x = self._display_aux(node.get_left())
            s = '%s' % node.get_key()
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if not node.get_left() and node.get_right():
            lines, n, p, x = self._display_aux(node.get_right())
            s = '%s' % node.get_key()
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self._display_aux(node.get_left())
        right, m, q, y = self._display_aux(node.get_right())
        s = '%s' % node.get_key()
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def left_then_right(self, node):
        node = self.left_rotate(node)
        print("first rotate \n")
        self.display(self.root)
        node = self.right_rotate(node.get_parent())
        print("second rotate \n")
        self.display(self.root)
        return node

    def right_then_left(self, node):
        node = self.right_rotate(node)
        print("first rotate")
        self.display(self.root)
        node = self.left_rotate(node.get_parent())
        print("second rotate")
        self.display(self.root)
        return node

    """deletes node from the dictionary

    @type node: AVLNode
    @pre: node is a real pointer to a node in self
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, node):
        left = node.get_left()
        right = node.get_right()
        parent = node.get_parent()
        if node == self.root:
            self.delete_root(node)
            return 0
        else:
            if not left.is_real_node() and right.is_real_node():  # no left child
                if node.key < parent.key:
                    right.set_parent(parent)
                    parent.set_left(right)
                else:
                    right.set_parent(parent)
                    parent.set_right(right)
            if not right.is_real_node() and left.is_real_node(): # no right child
                left.set_parent(parent)
                parent.set_left(left)
            else: # both left and right children exist
                if not right.get_left().is_real_node(): # the right child is the minimal
                    parent.set_right(right.get_right())
                    parent.get_right().set_parent(parent)
                else:
                    temp_tree = right.get_left()
                    while temp_tree.get_left.is_real_node():  # getting minimum value from right subtree
                        temp_tree = temp_tree.get_left()
                    if temp_tree == right.get_left(): # minimum was the left child
                        if temp_tree.get_right().is_real_node():
                            parent.set_right(temp_tree.get_right)
                            parent.get_right().set_right(parent)
                            temp_tree.get_right().set_right(right.get_right())
                            temp_tree.set_left(temp_tree)
                        else:
                            parent.set_right(temp_tree)
                            temp_tree_parent = temp_tree.get_parent()
                            temp_tree.set_parent(parent)
                            temp_tree_parent.set_left(temp_tree.get_right())
                            temp_tree.set_right(right.get_right)

    def delete_root(self, node):
        left = node.get_left()
        right = node.get_right()
        if not left.is_real_node() and right.is_real_node(): # only right child, right child can be of size 1 only
            right.set_parent(None)
            self.root = right
        if left.is_real_node() and not right.is_real_node(): # only left child, left child can be of size 1 only
            left.set_parent(None)
            self.root = left
        else: # two children
            if not right.get_left().is_real_node(): # if the right node is the minimum
                self.root = right
                right.set_parent(None)
                right.set_left(left)
            else:
                temp_tree = right
                while temp_tree.get_left().is_real_node():  # getting minimum value from right subtree
                    temp_tree = temp_tree.get_left()
                temp_tree_parent = temp_tree.get_parent()
                temp_tree.set_parent(None)
                self.root = temp_tree
                temp_tree_parent.set_left(temp_tree.get_right())
                temp_tree.set_left(left)
                temp_tree.set_right(temp_tree_parent)

        return


    """returns an array representing dictionary 

    @rtype: list
    @returns: a sorted list according to key of touples (key, value) representing the data structure
    """

    def avl_to_array(self):
        return None

    """returns the number of items in dictionary 

    @rtype: int
    @returns: the number of items in dictionary 
    """

    def size(self):
        return self.get_root().get_size()

    """splits the dictionary at a given node

    @type node: AVLNode
    @pre: node is in self
    @param node: The intended node in the dictionary according to whom we split
    @rtype: list
    @returns: a list [left, right], where left is an AVLTree representing the keys in the 
    dictionary smaller than node.key, right is an AVLTree representing the keys in the 
    dictionary larger than node.key.
    """

    def split(self, node):
        return None

    """joins self with key and another AVLTree

    @type tree: AVLTree 
    @param tree: a dictionary to be joined with self
    @type key: int 
    @param key: The key separting self with tree
    @type val: any 
    @param val: The value attached to key
    @pre: all keys in self are smaller than key and all keys in tree are larger than key,
    or the other way around.
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def join(self, tree, key, val):
        return None

    """compute the rank of node in the self

    @type node: AVLNode
    @pre: node is in self
    @param node: a node in the dictionary which we want to compute its rank
    @rtype: int
    @returns: the rank of node in self
    """

    def rank(self, node):
        r = node.left.size + 1
        y = node
        while y.is_real_node():
            if y.get_key() == y.get_parent().get_right().get_key():
                r += y.parent.left.size + 1
            y = y.parent
        return r

    """finds the i'th smallest item (according to keys) in self

    @type i: int
    @pre: 1 <= i <= self.size()
    @param i: the rank to be selected in self
    @rtype: int
    @returns: the item of rank i in self
    """

    def select(self, node, i):
        r = self.root.get_size + 1
        if r == i:
            return self.root
        elif i < r:
            return self.select(node, i)
        else:
            return self.select(node, i - r)

    """returns the root of the tree representing the dictionary

    @rtype: AVLNode
    @returns: the root, None if the dictionary is empty
    """

    def get_root(self):
        return self.root


firstTree = AVLTree()
nums = [1, 3, 5, 7, 9]
for num in nums:
    root = firstTree.insert(num, 1)
firstTree.delete(firstTree.root.get_right())
firstTree.display(firstTree.root)
