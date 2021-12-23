import time
from graph2 import *




i1 = Input(name='i1')
i2 = Input(name='i2')
x1 = Variable(1, name='x1')
x2 = Variable(2, name='x2')
c1 = Const(10, name='c1')

n2 = Calculate([i1, i2, x1, c1], name='n2')
n1 = Calculate([i1, x1], name='n1')
n4 = Calculate([n2, x2, c1], name='n4')
n3 = Calculate([n1, n2], name='n3')
n5 = Calculate([n3, n4, c1], name='n5')

n6 = Calculate([n5], name='n6')


graph = Graph([i1, i2], n6)
print(graph.nodes)
print(graph.variables)

graph.call([1, 1])

