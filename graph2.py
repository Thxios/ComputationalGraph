import numpy as np
from typing import *
from collections import deque


def as_list(target):
    if not isinstance(target, (list, tuple)):
        return [target]
    return list(target)


class Outbound:
    prefix = 'Outbound'

    def __init__(self, name=None):
        self.name = name
        self.n_outbounds = 0

        self.outbounds: List[Inbound_FwFeedFed_BwFeed] = []

    def __repr__(self):
        return '<{}{}>'.format(self.prefix, ' '+self.name if self.name else '')

    @property
    def forward_output(self):
        return None

    def add_outbound(self, outbound):
        assert outbound not in self.outbounds

        self.outbounds.append(outbound)
        self.n_outbounds += 1


class Input(Outbound):
    prefix = 'Input'
    def __init__(self, name=None):
        super(Input, self).__init__(name)
        self._assigned = False
        self._input = None

    def assign(self, value):
        # assert not self._assigned
        self._input = value

    @property
    def forward_output(self):
        # assert self._assigned
        return self._input


class Const(Outbound):
    prefix = 'Const'
    def __init__(self, value, name=None):
        super(Const, self).__init__(name)
        self._value = value

    @property
    def forward_output(self):
        return self._value


class BwFed(Outbound):
    prefix = ''
    def __init__(self, name=None):
        super(BwFed, self).__init__(name)

        self._backwarded_ids = {}

        self._backward_input = None

    @property
    def backwarded(self):
        return all(self._backwarded_ids.values())

    def reset_backwarded(self):
        for outbound_id in self._backwarded_ids:
            self._backwarded_ids[outbound_id] = False
        self._backward_input = None

    def add_outbound(self, outbound):
        super(BwFed, self).add_outbound(outbound)

        self._backwarded_ids[id(outbound)] = False

    # def backward_fed(self, dy, outbound):
    def backward_fed(self, outbound):
        assert not self._backwarded_ids[id(outbound)]

        self._backwarded_ids[id(outbound)] = True

    def _get_backward_input_from_outbounds(self):
        assert self.backwarded

        for outbound in self.outbounds:
            self._backward_input += outbound.backward_output(self)


class Variable(BwFed):
    prefix = 'Var'
    def __init__(self, initial_value, name=None):
        super(Variable, self).__init__(name)
        self._value = initial_value

    @property
    def forward_output(self):
        return self._value


class Inbound_FwFeedFed_BwFeed(BwFed):
    def __init__(self, inbounds, name=None):
        super(Inbound_FwFeedFed_BwFeed, self).__init__(name)

        self.inbounds: List[Union[Inbound_FwFeedFed_BwFeed, Outbound]] = inbounds
        self.n_inbounds = len(self.inbounds)

        self._n_bw_inbounds = 0
        self._bw_inbounds: List[Inbound_FwFeedFed_BwFeed] = []
        self._bw_inbound_indices: Dict[int, int] = {}
        self._forwarded_ids: Dict[int, bool] = {}
        self._backward_output = []

        for inbound in inbounds:
            inbound.add_outbound(self)

            if isinstance(inbound, Inbound_FwFeedFed_BwFeed):
                self._bw_inbounds.append(inbound)
                self._forwarded_ids[id(inbound)] = False

                self._backward_output.append(None)
                self._bw_inbound_indices[id(inbound)] = self._n_bw_inbounds
                self._n_bw_inbounds += 1

        self._forward_inputs = [None for _ in range(self.n_inbounds)]
        self._forward_output = None

    @property
    def forwarded(self):
        return all(self._forwarded_ids.values())

    @property
    def forward_output(self):
        # assert self.forwarded
        return self._forward_output

    def backward_output(self, inbound):
        assert self.backwarded
        return self._backward_output[self._bw_inbound_indices[id(inbound)]]

    def reset_forwarded(self):
        for inbound_id in self._forwarded_ids:
            self._forwarded_ids[inbound_id] = False

    def forward_fed(self, inbound):
        assert not self._forwarded_ids[id(inbound)]

        self._forwarded_ids[id(inbound)] = True

    def calc_forward(self):
        assert self.forwarded

        self._get_forward_input_from_inbounds()
        self._forward_output = self.call(*self._forward_inputs)

        for outbound in self.outbounds:
            outbound.forward_fed(self)

    def calc_backward(self):
        assert self.backwarded

        self._get_backward_input_from_outbounds()
        self._backward_output = self.derivative(self._backward_input, *self._forward_inputs)

        for inbound in self._bw_inbounds:
            inbound.backward_fed(self)

    def _get_forward_input_from_inbounds(self):
        assert self.forwarded

        for i, inbound in enumerate(self.inbounds):
            self._forward_inputs[i] = inbound.forward_output

    def call(self, *args):
        raise NotImplementedError

    def derivative(self, *args):
        raise NotImplementedError


class Calculate(Inbound_FwFeedFed_BwFeed):
    prefix = 'Calc'

    def call(self, *args):
        print(self)
        return 0


class Graph:
    def __init__(self, inputs, outputs):
        self.inputs: List[Input] = as_list(inputs)
        self.outputs: List[Inbound_FwFeedFed_BwFeed] = as_list(outputs)

        self.nodes: List[Inbound_FwFeedFed_BwFeed] = []
        self.variables: List[Variable] = []

        node_queue: Deque[Inbound_FwFeedFed_BwFeed] = deque()
        node_queue.extend(reversed(self.outputs))
        # seen_node_ids = set()

        while node_queue:
            current = node_queue.popleft()
            # current_id = id(current)
            # if current_id in seen_node_ids:
            #     print('already seen', current)
            #     continue
            # seen_node_ids.add(current_id)

            self.nodes.append(current)

            print(current, 'start inbound search')
            for inbound in reversed(current.inbounds):
                print(inbound)
                if isinstance(inbound, BwFed):
                    inbound.backward_fed(current)
                    if inbound.backwarded:
                        inbound.reset_backwarded()
                        if isinstance(inbound, Inbound_FwFeedFed_BwFeed):
                            node_queue.append(inbound)
                        elif isinstance(inbound, Variable):
                            self.variables.append(inbound)

            print(' ')
        self.nodes = list(reversed(self.nodes))
        self.variables = list(reversed(self.variables))


    def call(self, inputs):
        inputs = as_list(inputs)

        for i, input_ in enumerate(inputs):
            self.inputs[i].assign(input_)

        for node in self.nodes:
            node.calc_forward()
            node.reset_forwarded()

        outputs = []
        for output in self.outputs:
            outputs.append(output.forward_output)

        return outputs




