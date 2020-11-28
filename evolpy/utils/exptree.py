from inspect import signature, isfunction, getargspec
from random import choice, uniform
from math import floor

from evolpy import num_digits
from evolpy.utils.node import Node

'''
Class to implement expression trees
Each tree is composed by nodes which contains children.
All the non-leaf nodes contains operators and each leaf 
contain a one-dimensional array of real values.
'''
class ExpressionTree:
  '''
  Constructor method of class ExpressionTree
  --- Required args ---
  max_height: max height of the tree.
              It must be an integer.
  leaf_bounds:  dictionary of tree variables. 
                It must contain the name and limits in the format "name: (lower, upper)". 
  operators:  All operators that a node can contain. 
              It must be on an unstructured list.
  --- Optional args ---
  norm_leaf:  dictionary of normalizing function to each leaf.
              Is used to ensure that the tree will generate valid values.
              It must contain the name and function in the format "name: function pointer". 
  '''
  def __init__(self, max_height, leaf_bounds, norm_leaf=None, *operators):
    self.operators = []
    self.max_height = max_height
    self.leaf_bounds = leaf_bounds
    self.norm_leaf = norm_leaf
    self.addOp(operators)
    self.leafSize = len(list(leaf_bounds.keys()))
    self.root = None
    self.op_dict = { # used by stringify method
      'add': '+',
      'sub': '-',
      'mul': '*',
      'truediv': '/',
      'neg': '(-1)*',
    }
    
  def __str__(self):
    return self.root.__str__()
  
  def __len__(self):
    return len(self.root)
  
  def __iter__(self):
    return self.root.__iter__()
  
  def __getitem__(self, index):
    return self.root[index]
  
  def __setitem__(self, index, value):
    self.root[index] = value
    
  def __delitem__(self, index):
    del self.root[index]
  
  '''
  Method used to get the number of arguments that 
  the operator needs.
  '''
  def __num_args__(self, f):
    try:
      return len(signature(f).parameters)
    except:
      if isfunction(f):
        return len(getargspec(f).args)
      else:
        spec = f.__doc__.split('\n')[0]
        args = spec[spec.find('(')+1:spec.find(')')]
        return args.count(',')+1 if args else 0
  
  '''
  Method used to add operators to the tree.
  The argument "ops" must be an array with operators.
  '''
  def addOp(self, ops):
    self.operators.extend([{"op": op, "num_args": self.__num_args__(op)} for op in ops])
  
  '''
  Method used to create an expression tree.
  It will create a tree of max_height.
  Non-leaf nodes will have operators.
  The leaves will have arrays with uniform aleatory 
  values between the leaf boundaries divided by 10. 
  This value was used to try maintain the value outputed 
  ''' 
  def create(self, max_height=None, parent=None, normalize_by=1, curr_height=1):
    if not max_height:
      max_height = self.max_height
    if(curr_height != max_height):
      node = Node(choice(self.operators), parent=parent)
      for i in range(node.value["num_args"]):
        self.create(max_height, node, normalize_by, curr_height+1)
    else:
      node = Node([round(uniform(*v)/normalize_by, num_digits) for v in self.leaf_bounds.values()], parent=parent)
    return node
  
  def destroy(self):
    self.root = None
  
  '''
  Method used to obtain non-normalized evaluation 
  of each tree variable
  '''
  def eval(self):
    return dict(zip(list(self.leaf_bounds.keys()), self.__eval(self.root)))

  '''
  Method used to obtain normalized evaluation 
  of each tree variable
  '''
  def norm_eval(self):
    return {k: self.norm_leaf[k](v, *self.leaf_bounds[k]) for k, v in self.eval().items()}
  
  '''
  Method evaluates the tree by applying the leaves to the operators.
  '''
  def __eval(self, node):
    if(node.isLeaf()):
      return node.value
    else:
      children = [self.__eval(child) for child in node.children]
      val = []
      for i in range(self.leafSize):
        nums = [child[i] for child in children]
        try:
          calc = node.value['op'](*nums)
        except:
          calc = 1
        val.append(calc)
      return val

  '''
  Method to create a safe copy of an expression tree i.e. create a 
  new tree with the same nodes, but without pointer connexions 
  between the original and copied tree.
  '''
  def safecopy(self):
    ops = [x['op'] for x in self.operators]
    t = ExpressionTree(self.max_height, self.leaf_bounds, self.norm_leaf, *ops)
    t.root = self.root.safecopy()
    return t 

  '''
  Method used to get a string representation of the tree.
  '''
  def stringify(self, node=None, mask=False, variables={}):
    if not node:
      node = self.root
    if node.isLeaf():
      if not mask:
        return str(node.value)
      else:
        name = "x{}".format(len(variables.keys()))
        variables[name] = node.value
        return name, variables
    else:
      children = []
      for child in node.children:
        if not mask:
          ret = self.stringify(child, mask, variables)
        else:
          ret, _ = self.stringify(child, mask, variables)
        children.append(ret)
      try:
        op = self.op_dict[node.value['op'].__name__]
      except:
        op = node.value['op'].__name__
      if op == "log" or op == "sqrt":
        children = ["(abs({}))".format(child) for child in children]
      if len(children) > 1:
        ret = "({}){}({})".format(children[0], op, children[1])
      else:
        ret = "{}({})".format(op, children[0])
      if not mask:
        return ret
      else:
        return ret, variables
