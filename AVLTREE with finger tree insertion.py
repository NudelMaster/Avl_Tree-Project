# username - complete info
# id1      - complete info
# name1    - complete info
# id2      - complete info
# name2    - complete info


"""A class represnting a node in an AVL tree"""
import random
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
        self.rank = 0

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

    def get_rank(self):
        return self.rank

    def set_rank(self, rank):
        self.rank = rank

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

    """returns whether self is leaf or not

    @rtype: bool
    @returns: True if self is a leaf, False otherwise
    """

    def is_leaf(self):
        return not self.left.is_real_node() and not self.right.is_real_node()

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def is_real_node(self):
        return self.height > -1

    """returns the minimum node from the given node in the tree

    @rtype: AVLNode
    @returns: A node in the subtree with the minimum key
    """

    def min_node(self):
        temp = self
        while temp.get_left().is_real_node():
            temp = temp.get_left()
        return temp

    def max_node(self):
        temp = self
        while temp.get_right().is_real_node():
            temp = temp.get_right()
        return temp

    """returns the successor in an in-order method

    @rtype: AVLNode
    @returns: in order successor node
    """

    def in_order_successor(self):
        if self.right.is_real_node():
            return self.right.min_node()
        else:
            return self

    """Overwrites the previous height of the node with the current height

    @pre: self is AVLNode
    """

    def updateHeight(self):
        left_height = max(self.get_left().get_height(), 0)
        right_height = max(self.get_right().get_height(), 0)
        if self.is_leaf():
            self.set_height(0)
        else:
            self.set_height(1 + max(left_height, right_height))

    """Overwrites the previous size of the node with the current size

    @pre: self is AVLNode
    """

    def updateSize(self):
        self.set_size(1 + self.left.size + self.right.size)


"""
A class implementing an AVL tree.
"""


class AVLTree(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self, root=None):
        virtualNode = AVLNode(None, -1)
        self.root = root
        self.virtualNode = virtualNode
        self.min = None
        self.max = None

    # add your fields here

    """searches for a value in the dictionary corresponding to the key

    @type key: int
    @param key: a key to be searched
    @rtype: any
    @returns: the value corresponding to key.
    """

    def search(self, key):
        root = self.root

        def search_rec(node, key):
            if node is None or node.get_key() == key:
                return node
            if node.get_right().is_real_node() and node.get_key() < key:
                return search_rec(node.get_right(), key)
            if node.get_left().is_real_node():
                return search_rec(node.get_left(), key)

        return search_rec(root, key)

    """inserts recursively val to position i in the dictionary

    @type node: AVLNode
    @param node: AVLNode to be inserted to self
    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: any
    @param val: the value of the item
    @rtype: AVLNode
    """

    def insert_rec(self, node, key, val):  # regular insert to binary tree
        if key < node.get_key():
            node.set_rank(node.get_rank() + 1)
            if not node.get_left().is_real_node():
                node.set_left(AVLNode(key, val))
                node.get_left().set_parent(node)
                return node.get_left()
            return self.insert_rec(node.get_left(), key, val)
        elif key > node.get_key():
            if not node.get_right().is_real_node():
                node.set_right(AVLNode(key, val))
                node.get_right().set_parent(node)
                return node.get_right()
            return self.insert_rec(node.get_right(), key, val)

    """inserts val at position i in the dictionary

    @type key: int
    @pre: key currently does not appear in the dictionary
    @param key: key of item that is to be inserted to self
    @type val: any
    @param val: the value of the item
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def update_max(self):
        self.max = self.root.max_node()

    def max_insert_rec(self, node, key, val):
        if key < node.get_key():
            if node is self.root:  # if we have reached root
                return self.insert_rec(node, key, val)  # regular recursive insertion
            elif node.get_parent().get_right() is not node:  # if we are in the left subtree
                return self.insert_rec(node, key, val)
            elif key < node.get_parent().get_key():  # if we are in right subtree and parent's key is bigger then key
                return self.max_insert_rec(node.get_parent(), key, val)
            else:  # if key is bigger than parent
                return self.insert_rec(node, key, val)
        else:  # key > node.key e
            return self.insert_rec(node, key, val)

    def max_insert(self, key, val):
        global counter
        root = self.get_root()
        if root is None:
            self.root = AVLNode(key, val)
            self.root.set_left(AVLNode(None, -1))
            self.root.set_right(AVLNode(None, -1))
            self.root.get_left().set_parent(self.root)
            self.root.get_right().set_parent(self.root)
            self.root.updateHeight()
            self.root.set_size(1)
            self.max = self.root
            return 0, 0, 0
        rb_num = 0
        y = self.max_insert_rec(self.max, key, val)  # get inserted node
        y.set_left(AVLNode(None, -1))
        y.set_right(AVLNode(None, -1))
        y.updateHeight()
        y.updateSize()
        temp = self.max
        inserted = y
        if y.get_key() > self.max.get_key():
            self.max = y
        y = y.get_parent()
        while y:  # check if we the node is not null (happens after we get the parent of the root)
            y.updateSize()
            temp_height = y.get_height()
            y.updateHeight()
            balance_factor = self.BFS(y)
            if abs(balance_factor) < 2 and y.get_height() == temp_height:
                y = y.get_parent()
            elif abs(balance_factor) < 2 and y.get_height() != temp_height:
                y = y.get_parent()
                rb_num += 1
            elif abs(balance_factor) == 2:
                # print("\n Before rotate")
                # self.display(self.root)
                y, rb_num = self.rotate(y, balance_factor)
                y = y.get_parent()
        counter = self.search_from_max(temp, key)
        rank = self.rank(inserted)
        return rb_num, counter, rank

    def search_from_max(self, node, key):
        global counter
        if key < node.get_key():
            if node is self.root:  # if we have reached root
                return self.search_rec_max(node, key)  # regular recursive insertion
            elif node.get_parent().get_right() is not node:  # if we are in the left subtree
                return self.search_rec_max(node, key)
            elif key < node.get_parent().get_key():  # if we are in right subtree and parent's key is bigger then key
                counter += 1
                return self.search_from_max(node.get_parent(), key)
            elif key == node.get_parent().get_key():  # parent is the inserted node
                counter += 1
                return counter
            else:
                return self.search_rec_max(node, key)
        else:  # key > node.key e
            return self.search_rec_max(node, key)

    def search_rec_max(self, node, key):
        global counter
        if node.get_key() == key:
            return counter
        if node.get_right().is_real_node() and node.get_key() < key:
            counter += 1
            return self.search_rec_max(node.get_right(), key)
        elif node.get_left().is_real_node():
            counter += 1
            return self.search_rec_max(node.get_left(), key)

    """Calculates the balance factor of a node in the tree
    @type node: AVLNode
    @pre: node is not virtual
    @param node: node to calculate it's balance factor
    @rtype: int
    @returns: the balance factor of the given node
    """

    def BFS(self, node):
        return node.get_left().height - node.get_right().height

    """returns a tuple representing the rotation number and the rotated node after performing right or left or both rotations according to the balance factor
    @type node: AVLNode
    @pre: node is not virtual
    @param node: node that has it's BFS calculated and needs to be rotated
    @type BFS: int
    @pre: -2<= BFS <= 2
    @param node: the balance factor value according to the node that has already been calculated
    @rtype: tuple
    @returns: the node that has been rotated and the number of rotations
    """

    def rotate(self, node, BFS):
        rb_count = 0
        balance_factor = BFS
        if balance_factor >= 2:
            if self.BFS(node.get_left()) == -1:
                node = self.left_then_right(node.get_left())
                rb_count += 2
            else:
                node = self.right_rotate(node)
                rb_count += 1
        elif balance_factor <= -2:
            if self.BFS(node.get_right()) == 1:
                node = self.right_then_left(node.get_right())
                rb_count += 2
            else:
                node = self.left_rotate(node)
                rb_count += 1
        return node, rb_count

    """Performs a left rotation to the given node from the tree
    @type node: AVLNode
    @pre: node is not virtual
    @param node: node to be rotated to the left
    @rtype: AVLNode
    @returns: The right node of the given node that is being rotated to the left
    """

    def left_rotate(self, node):
        B = node
        A = B.get_right()
        B.set_right(A.get_left())
        B.get_right().set_parent(B)
        A.set_left(B)
        self.fix_parent(A, B)
        B.updateHeight()
        A.updateHeight()
        B.updateSize()
        A.updateSize()
        # print("\n left rotate")
        # self.display(self.root)
        return A

    """Performs a right rotation to the given node from the tree

    @type node: AVLNode
    @pre: node is not virtual
    @param node: node to be rotated to the right
    @rtype: AVLNode
    @returns: The left node of the given node that is being rotated to the right
    """

    def right_rotate(self, node):
        B = node
        A = B.get_left()
        B.set_left(A.get_right())
        B.get_left().set_parent(B)
        A.set_right(B)
        self.fix_parent(A, B)
        B.updateHeight()
        A.updateHeight()
        B.updateSize()
        A.updateSize()
        # print("\n right rotate")
        # self.display(self.root)
        return A

    """Performs left then right rotation to the given node from the tree

    @type node: AVLNode
    @pre: node is not virtual
    @param node: node to be rotated to the left and then to the right
    @rtype: AVLNode
    @returns: the current subtree from the given node after being rotated
    """

    def left_then_right(self, node):
        node = self.left_rotate(node)
        node = self.right_rotate(node.get_parent())
        return node

    """Performs right then left rotation to the given node from the tree

    @type node: AVLNode
    @pre: node is not virtual
    @param node: node to be rotated to the right and then to the left
    @rtype: AVLNode
    @returns: the current subtree from the given node after being rotated
    """

    def right_then_left(self, node):
        node = self.right_rotate(node)
        node = self.left_rotate(node.get_parent())
        return node

    """Fixes references to the parents of the nodes after rotation
    @type node: AVLNode
    @pre: node is not virtual
    @param node: node that it's current parent is not updated after rotation
    @type node: AVLNode
    @pre: node is not virtual
    @param node: node that it's current parent is not updated after rotation
    """

    def fix_parent(self, A, B):
        A.set_parent(B.get_parent())
        if B == self.root:
            self.root = A
        elif B.get_parent().get_left().get_key() == B.get_key():
            A.get_parent().set_left(A)
        else:
            A.get_parent().set_right(A)
        B.set_parent(A)

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

    """deletes node from the dictionary

    @type node: AVLNode
    @pre: node is a real pointer to a node in self
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, node):
        left = node.get_left()
        right = node.get_right()
        if node.is_leaf():
            y = node.get_parent()
        elif not right.is_real_node():  # only left subtree exists
            y = left  # left becomes the parent of the deleted node
        else:
            successor = node.in_order_successor()  # get successor (go all left from right subtree)
            successor_parent = successor.get_parent()
            if successor_parent is node:
                y = successor
            else:
                y = successor_parent
        self.root = self.BTS_delete(self.root, node.get_key())  # Regular deletion from BST
        if not self.root.is_real_node():  # if we get an empty tree, after deleting the root of size 1
            self.root = None
            return 0
        rb_num = 0
        while y:
            y.size -= 1
            temp_height = y.get_height()
            y.updateHeight()
            y.updateSize()
            balance_factor = self.BFS(y)
            if abs(balance_factor) < 2 and y.get_height() == temp_height:
                y = y.get_parent()
            elif abs(balance_factor) < 2 and y.get_height() != temp_height:
                y = y.get_parent()
                rb_num += 1
            elif abs(balance_factor) >= 2:
                print("\n Before rotate")
                self.display(self.root)
                y, rb_num = self.rotate(y, balance_factor)
                y = y.get_parent()
                print("\n After rotate")
                self.display(self.root)
        return rb_num

    """Performs a regular deletion in a binary search tree
    @type node: AVLNode
    @pre: node exists in self
    @param node: root of the tree that needs the node with the given key to be deleted 
    @type key: int
    @pre: key is not None
    @param key: the key of the node to be deleted from the tree
    @rtype: AVLNode
    @returns: the root of the tree after the deletion of the node with the given key
    """

    def BTS_delete(self, node, key):  # recursive method to delete node with given key
        if not node.is_real_node():
            return None
        if key < node.get_key():
            node.set_left(self.BTS_delete(node.get_left(), key))
            node.get_left().set_parent(node)
            return node
        elif key > node.get_key():
            node.set_right(self.BTS_delete(node.get_right(), key))
            node.get_right().set_parent(node)
            return node

        # After reaching the node starting the deletion
        if node.is_leaf():
            return AVLNode(None, -1)

        if not node.get_left().is_real_node():
            temp = node.get_right()
            return temp
        elif not node.get_right().is_real_node():
            temp = node.get_left()
            return temp

        successor = node.in_order_successor()
        successor_parent = successor.get_parent()

        if successor_parent is not node:
            successor_parent.set_left(successor.get_right())
            successor_parent.get_left().set_parent(successor_parent)
        else:
            successor_parent.set_right(successor.get_right())
            successor_parent.get_right().set_parent(successor_parent)

        node.set_key(successor.get_key())
        node.set_value(successor.set_value(successor.get_value()))

        return node

    """returns an array representing dictionary 

    @rtype: list
    @returns: a sorted list according to key of touples (key, value) representing the data structure
    """

    def avl_to_array(self):
        def avl_to_array_rec(array, node):
            if node.is_real_node:
                avl_to_array_rec(array, node.left)
                array.append(node)
                avl_to_array_rec(array, node.right)

        ret_list = []
        avl_to_array_rec(ret_list, self.root)
        return ret_list

    """returns the number of items in dictionary 

    @rtype: int
    @returns: the number of items in dictionary 
    """

    def size(self):
        return self.root().get_size()

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
        left = AVLTree(node.left)
        right = AVLTree(node.right)
        father = node.parent()
        while father.is_real_node:
            if father.right is node:
                AVLTree(father.left).join(left, father.key, father.value)
            else:
                AVLTree(father.right).join(right, father.key, father.value)
            node = father
            father = node.parent
        return left, right

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
        def join(self, tree, key, val):

            retval = abs(self.root.hight - tree.root.hight) + 1
            if self.get_root().get_key < key:
                self.rec_join(tree, key, val)
            else:
                tree.rec_join(self, key, val)
            return retval

    def UnbalancedJoin(self, tree, key, val):
        node = AVLNode(key, val)
        node.left = self.root
        node.right = tree.root
        self.root.set_parent(node)
        tree.root.set_parent(node)
        return node

    def rec_join(self, big, key, val):
        if abs(self.root.get_hight() - big.get_root().get_hight()) <= 1:
            new_root = self.UnbalancedJoin(big, key, val)
            new_root.set_height(max(self.root.get_hight(), big.get_root().get_hight()) + 1)
            self.root = new_root
        elif self.root.hight > big.root.hight:
            new_root = AVLTree(self.root.get_right()).rec_join(big, key, val)
            self.root.set_right(new_root)
            new_root.set_parent(self.root)
            self.root = self.Balance(new_root)
        else:
            new_root = AVLTree(self.root.get_left()).rec_join(big, key, val)
            self.root.set_left(new_root)
            new_root.set_parent(self.root)
            self.root = self.Balance(new_root)
        return self.root

    def Balance(self, node):
        true_root = node
        while node.is_real_node():
            bfs = self.BFS(node)
            (true_root, rotate_num) = self.rotate(node, bfs)
            if rotate_num != 0:
                break
            node = true_root.get_parent()
        return true_root

    """compute the rank of node in the self

    @type node: AVLNode
    @pre: node is in self
    @param node: a node in the dictionary which we want to compute its rank
    @rtype: int
    @returns: the rank of node in self
    """

    def rank(self, node):
        r = node.left.size
        y = node
        while y is not self.root:
            if y is y.get_parent().get_right():
                r += y.parent.left.size + 1
            y = y.get_parent()
        return r

    """finds the i'th smallest item (according to keys) in self

    @type i: int
    @pre: 1 <= i <= self.size()
    @param i: the rank to be selected in self
    @rtype: int
    @returns: the item of rank i in self
    """

    def select(self, node, i):
        r = self.root.get_size()
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


def list_builder(i, lst_type):
    n = pow(2, i) * 1500
    return_lst = []
    while n > 0:
        return_lst.append(n)
        n -= 1
    if lst_type == "reverse":
        return return_lst
    elif lst_type == "shuffle":
        random.shuffle(return_lst)
        return return_lst
    elif lst_type == "almost":
        return_lst.reverse()
        split_lst = [return_lst[i: i + 300] for i in range(0, len(return_lst), 300)]
        return_lst = []
        for lst in split_lst:
            lst.reverse()
            return_lst = return_lst + lst
        return return_lst
    else:
        return None


nums_to_insert = [3, 2, 5, 1, 4]

firstTree = AVLTree()
nums = list(range(50))
random.shuffle(nums)
# nums.reverse()
sum_search = 0
sum_transpositions = 0
sum_balance = 0
for num in nums:
    counter = 0
    print("inserting", num)
    tup = firstTree.max_insert(num, 1)

    print("number of balancing", tup[0], "number of searches ", tup[1], "\n")
    print("\n", "rank of", num, "is", tup[2], "\n")
    sum_transpositions += tup[2]
    sum_search += tup[1]
    sum_balance += tup[0]
firstTree.display(firstTree.root)
print("\n", "size of root is", firstTree.root.size)
print("\n", "cost of sorting is", sum_search + sum_balance)
print("\n", "number of transpositions in original array is", sum_transpositions)
print("\n", "number of rebalancing in total is", sum_balance)
# firstTree.update_max()
#
# firstTree.max_insert(8, 1)
# firstTree.display(firstTree.root)

# # 12 problematic deletion
# deleted_nums = [6,3,10,4,16,13,0,11,9,7,18,5,19,2,14,12,1,8,15,17]
# for num in deleted_nums:
#     node_to_delete = firstTree.search(num)
#     print("deleting", num)
#     firstTree.delete(node_to_delete)
#     if firstTree.root:
#         #print("minimum is", firstTree.min.get_key())
#         #print("maximum is", firstTree.max.get_key())
#         firstTree.display(firstTree.root)

while firstTree.root:
    random_key = random.choice(nums)
    node_to_delete = firstTree.search(random_key)
    print("deleting", random_key)
    firstTree.delete(node_to_delete)
    if firstTree.root:
        firstTree.display(firstTree.root)
    else:
        print(None)
    nums.remove(random_key)

# firstTree.delete(firstTree.root.get_right())
# firstTree.display(firstTree.root)
