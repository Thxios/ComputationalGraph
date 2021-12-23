
import numpy as np
from typing import *


def num2str(num):
    return '({})'.format(num) if num < 0 else str(num)

class Node:
    prefix = 'Node'

    def __init__(self, inbounds, name=''):
        self.name = self.prefix + str(name)

        self.n_inbounds = len(inbounds)
        self.n_outbounds = 0

        self._inbounds = dict()
        self._forward_ready = dict()
        self._fed_forward_init: List[Union[None, float]] = \
            [None for _ in range(self.n_inbounds)]
        for i, inbound in enumerate(inbounds):
            if isinstance(inbound, (int, float)):
                self._fed_forward_init[i] = inbound
                continue

            inbound.connect_outbound(self)
            self._inbounds[inbound] = i
            self._forward_ready[inbound] = False

        self._forward_inputs = self._fed_forward_init[:]
        self.forward_output = None

        self._outbounds = dict()
        self._backward_ready = dict()
        self._backward_input = 0
        self.backward_outputs = []

    def __repr__(self):
        return '<{}>'.format(self.name)

    def connect_outbound(self, outbound):
        assert outbound not in self._outbounds

        self._outbounds[outbound] = self.n_outbounds
        self._backward_ready[outbound] = False
        self.backward_outputs.append(None)
        self.n_outbounds += 1

    def reset(self):
        self._forward_inputs = self._fed_forward_init[:]
        for inbound in self._inbounds:
            self._forward_ready[inbound] = False
        self.forward_output = None

        self._backward_input = 0
        for outbound in self._outbounds:
            self._backward_ready[outbound] = False
            self.backward_outputs[self._outbounds[outbound]] = None

    def forward(self, x, inbound=None):
        if inbound is not None:
            self._forward_ready[inbound] = True
            self._forward_inputs[self._inbounds[inbound]] = x

        if all(self._forward_ready.values()):

            self.forward_output = self.call(*self._forward_inputs)
            # print(self, 'Forward', self.symbol(*self._forward_inputs), '->', self.forward_output)
            for outbound in self._outbounds:
                # outbound.forward_out(self.forward_output, self)
                outbound.forward(self.forward_output, self)

    def backward(self, dy, outbound=None):
        if outbound is not None:
            self._backward_ready[outbound] = True
        self._backward_input += dy

        if all(self._backward_ready.values()):
            self.backward_outputs = self.derivative(*self._forward_inputs)
            if self.n_inbounds == 1:
                self.backward_outputs = (self.backward_outputs,)

            # print(self, 'Backward', self._inbounds, self.backward_outputs, self._backward_input)
            for inbound in self._inbounds:
                inbound.backward(self._backward_input * self.backward_outputs[self._inbounds[inbound]], self)

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def derivative(self, *args, **kwargs):
        raise NotImplementedError

    def symbol(self, *args, **kwargs):
        return ''

class Input:
    prefix = 'Input'
    def __init__(self, name=''):
        self.name = self.prefix + str(name)
        self._outbounds: List[Node] = []
        self.forward_input = None
        self._backward_ready = dict()
        self.backward_input = 0

    def __repr__(self):
        return '<{}>'.format(self.name)

    def connect_outbound(self, outbound):
        self._outbounds.append(outbound)

    def forward(self, x):
        print(self, x)
        for outbound in self._outbounds:
            outbound.forward(x, self)

    def backward(self, dy, outbound):
        if outbound is not None:
            self._backward_ready[outbound] = True
        self.backward_input += dy

        if all(self._backward_ready.values()):
            # print(self, 'Backward', self.backward_input)
            pass

class Add(Node):
    prefix = 'Add'
    def __init__(self, a, b, name=''):
        Node.__init__(self, [a, b], name=name)
    def call(self, a, b):
        return a + b
    def derivative(self, a, b):
        return 1, 1
    def symbol(self, a, b):
        return '{}+{}'.format(num2str(a), num2str(b))

class Subtract(Node):
    prefix = 'Sub'
    def __init__(self, a, b, name=''):
        Node.__init__(self, [a, b], name=name)
    def call(self, a, b):
        return a - b
    def derivative(self, a, b):
        return 1, -1
    def symbol(self, a, b):
        return '{}-{}'.format(num2str(a), num2str(b))

class Multiply(Node):
    prefix = 'Mul'
    def __init__(self, a, b, name=''):
        Node.__init__(self, [a, b], name=name)
    def call(self, a, b):
        return a * b
    def derivative(self, a, b):
        return b, a
    def symbol(self, a, b):
        return '{}×{}'.format(num2str(a), num2str(b))

class Divide(Node):
    prefix = 'Div'
    def __init__(self, a, b, name=''):
        Node.__init__(self, [a, b], name=name)
    def call(self, a, b):
        return a / b
    def derivative(self, a, b):
        return 1/b, -a / b**2
    def symbol(self, a, b):
        return '{}/{}'.format(num2str(a), num2str(b))

class Power(Node):
    prefix = 'Pow'
    def __init__(self, a, b, name=''):
        Node.__init__(self, [a, b], name=name)
    def call(self, a, b):
        assert a > 0
        return a ** b
    def derivative(self, a, b):
        return b * a**(b - 1), np.log(a) * a**b
    def symbol(self, a, b):
        return '{}^{}'.format(num2str(a), num2str(b))

class Inverse(Node):
    prefix = 'Inv'
    def __init__(self, x, name=''):
        Node.__init__(self, [x], name=name)
    def call(self, x):
        assert x != 0
        return 1 / x
    def derivative(self, x):
        return -1 / x**2
    def symbol(self, x):
        return '1/{}'.format(num2str(x))

class Exponential(Node):
    prefix = 'Exp'
    def __init__(self, x, name=''):
        Node.__init__(self, [x], name=name)
    def call(self, x):
        return np.exp(x)
    def derivative(self, x):
        return np.exp(x)
    def symbol(self, x):
        return 'e^{}'.format(num2str(x))

class Negative(Node):
    prefix = 'Neg'
    def __init__(self, x, name=''):
        Node.__init__(self, [x], name=name)
    def call(self, x):
        return -x
    def derivative(self, x):
        return -1
    def symbol(self, x):
        return '-{}'.format(num2str(x))

class Graph:
    def __init__(
            self,
            inputs: Union[Input, List[Input]],
            outputs: Union[Node, List[Node]]
    ):
        if isinstance(inputs, (Input, Node)):
            self.inputs = [inputs]
        else:
            self.inputs = inputs
        self.n_inputs = len(self.inputs)

        if isinstance(outputs, Node):
            self.outputs = [outputs]
        else:
            self.outputs = outputs
        self.n_outputs = len(self.outputs)

    def __call__(self, inputs):
        if self.n_inputs == 1:
            self.inputs[0].forward(inputs)
        else:
            for i in range(self.n_inputs):
                self.inputs[i].forward(inputs[i])

        if self.n_outputs == 1:
            return self.outputs[0].forward_output
        else:
            return [output.forward_output for output in self.outputs]

    def backpropagate(self):
        for output in self.outputs:
            output.backward(1)

        return [input_.backward_input for input_ in self.inputs]


if __name__ == '__main__':
    x = Input('X')
    y = Input('Y')

    output = Divide(Add(Exponential(x), Multiply(x, Exponential(y))), Power(x, 2))

    graph = Graph([x, y], output)

    print(graph((2, 1)))
    print('\n')
    print(graph.backpropagate())



