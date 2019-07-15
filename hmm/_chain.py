import numpy as np

from _base import HMM, Memoize

class Chain(HMM):
    """
    y = observed
    x = hidden
    phi = data term
    psi = smoothness term
    """

    def __init__(self, length, phi, psi, possible_values=[-1, 1]):

         self._len = length
         self._phi = phi
         self._psi = psi
         self._possible_values = possible_values

         self._observed = None

    def _update_observed(self, observed):
        self._observed = observed

    def _forward_pass(self, method):
        messages = {}
        msg_left = lambda x_1: 1 # no information traveling left to x1

        for i in range(1, self._len):
            y_left = self._observed[i - 1]
            msg = self._belief_propagation(msg_left, y_left, method)
            messages[(i, i + 1)] = msg
            msg_left = msg

        return messages

    def _backward_pass(self, method):
        messages = {}
        msg_right = lambda x_n: 1 # no information traveling right to x_n

        for i in range(self._len, 1, -1):
            y_right = self._observed[i - 1]
            msg = self._belief_propagation(msg_right, y_right, method)
            messages[(i, i - 1)] = msg
            msg_right = msg

        return messages

    def _belief_propagation(self, msg_before, y_before, method):
        agg_func = {
            'sum_product': sum,
            'max_product': max
        }[method]

        msg_next = lambda x_i: agg_func([
            self._phi(x_before, y_before) * self._psi(x_before, x_i) * msg_before(x_before)
            for x_before in self._possible_values
        ])

        return Memoize(msg_next)

    def _get_beliefs(self, method):
        forward_msgs = self._forward_pass(method)
        backward_msgs = self._backward_pass(method)

        beliefs = {}
        for i in range(self._len):
            y_i = self._observed[i]
            msg_left = forward_msgs.get((i , i + 1), lambda x_i: 1)
            msg_right = backward_msgs.get((i + 2 , i + 1), lambda x_i: 1)

            belief_func = lambda x_i: self._phi(x_i, y_i) * msg_left(x_i) * msg_right(x_i)
            z_norm = sum([belief_func(v) for v in self._possible_values])

            beliefs[i + 1] = [(1 / z_norm) * belief_func(x_i) for x_i in self._possible_values]

        return beliefs

    def _get_marginal_beliefs(self):
        return self._get_beliefs(method='sum_product')

    def _get_max_apostriori_beliefs(self):
        beliefs = self._get_beliefs(method='max_product')
        return [self._possible_values[np.argmax(b)] for k, b in beliefs.items()]
