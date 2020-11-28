from functools import reduce
from itertools import chain
from copy import copy

'''
Class to implement trees
Each tree is composed of nodes that contain children.
'''
class Node:
  '''
  Constructor method of class Node
  --- Required args ---
  value: The value that the node will store.
  --- Optional args ---
  parent: The parent node of the node being created. 
          If set, it will call the 'insert' method of the parent node. 
  '''
  def __init__(self, value, parent=None):
    self.value = value 
    self.children = []
    self.level = parent.level + 1 if parent else 0
    self.parent = parent
    if(parent):
      parent.insert(self)
      
  '''
  Used to give a human-readable representation 
  of the node. As this class was written to be used with 
  expression trees that may contain operators on the nodes, 
  this method also handles this, obtaining the operator name.
  '''
  def __str__(self):
    try: 
      ret = "\t"*self.level
      try:
        ret += repr(self.value if self.isLeaf() else self.value['op'].__name__)
      except:
        ret += repr(self.value)
      ret += "\n"
      for child in self.children:
          ret += child.__str__()
      return ret
    except:
      print(self.__dict__)
      raise
  
  '''
  Get the number of nodes in the tree, including the root.
  '''
  def __len__(self):
    return reduce(lambda x,y: x+y.__len__(), self.children, 1)
  
  def __iter__(self):
    yield self
    for child in chain.from_iterable(map(iter, self.children)):
      yield child

  def __getitem__(self, index, aux=0):
    if type(index) != int:
      raise TypeError("Node indices must be integers, not {}".format(type(index).__name__))
    if index < 0:
      index += len(self)
      if index < 0:
        raise IndexError("index {} out of range".format(index))
    if index == aux:
      return self
    for i, child in enumerate(self.children):
      if i == 0:
        offset = 1
      else:
        cont = 1 + len(self.children[0])
        offset = cont if cont > 1 else 2
      item = child.__getitem__(index, aux+offset)
      if item:
        return item
    if aux != 0:
      return None
    else:
      raise IndexError("index {} out of range".format(index))
  
  '''
  Used to update the level of a node and its children.
  '''   
  def __updatelevel__(self, level):
    try:
      self.level = level
    except:
      print(self)
      print(self.__dict__)
      print(self.parent.__dict__)
      print(self.parent)
      raise UpdateError("Critical error when updating level of the node")
    for child in self.children:
      child.__updatelevel__(level+1)
  

  def __setitem__(self, index, value):
    if(index == 0):
      raise IndexError("index must be greater than 0.\n" +
                       "To assign to the root of a tree, " +
                       "make it directly, without index")
    if value:
      if(type(value) != type(self)):
        raise TypeError("Type of assigned item must be Node, not {}".format(type(value).__name__))
      item = self[index]
      parent = item.parent
      value.parent = parent
      value.__updatelevel__(item.level)
      parent.children.remove(item)
      parent.children.append(value)
    else:
      item = self[index]
      parent = item.parent
      parent.children.remove(item)
         
  def __delitem__(self, index):
    item = self[index]
    if item.parent:
      item.parent.children.remove(item)
    else:
      raise IndexError("index must be greater than 0.\n" +
                       "To delete the root of a tree, call del " +
                       "without the index or make it points to None")
  
  '''
  Method to insert a child in a node, considering a binary tree.
  '''
  def insert(self, node):
    if(len(self.children) < 2):
      self.children.append(node)   
    else:
      print("Node Full")
  
  '''
  Method to check if a node is a leaf.
  '''
  def isLeaf(self):
    return len(self.children) == 0

  '''
  Method used to get the number of leaves in the tree
  '''
  def num_leaves(self, node=None):
    if node is None:
      node = self
    return 1 if node.isLeaf() \
           else reduce(lambda x,y: x + self.__num_leaves__(y), self.children, 0)

  '''
  Method to create a safe copy of a node i.e. create a new node 
  with the same value and children, but without pointer connexions 
  between the original and copied node.
  '''
  def safecopy(self, ind=None, parent=None):
    if not ind:
      ind = self
    newInd = Node(copy(ind.value), parent)
    for child in ind.children:
      self.safecopy(child, newInd)
    return newInd
