import numpy as np

from _base import HMM

class Chain(HMM):
    """
    y = observed
    x = hidden
    phi = data term
    psi = smoothness term
    """

    def __init__(self, length, phi, psi, possible_values=[0, 1]):

         self._len = length
         self._phi = phi
         self._psi = psi
         self._possible_values = possible_values

         self._observed = None

    def _unit_msg(self):
        unit_msg = dict((k, 1) for k in self._possible_values)
        return self._normalize_msg(unit_msg)

    def _update_observed(self, observed):
        self._observed = observed

    def _forward_pass(self, method):
        messages = {}
        msg_left = self._unit_msg() # no information traveling left to x1

        for i in range(1, self._len):
            y_left = self._observed[i - 1]
            msg = self._belief_propagation(msg_left, y_left, method)
            messages[(i, i + 1)] = msg
            msg_left = msg

        return messages

    def _backward_pass(self, method):
        messages = {}
        msg_right = self._unit_msg() # no information traveling right to x_n

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
            self._phi(x_before, y_before) * self._psi(x_before, x_i) * msg_before[x_before]
            for x_before in self._possible_values
        ])

        msg_next = dict((k, msg_next(k)) for k in self._possible_values)
        return self._normalize_msg(msg_next)

    def _get_beliefs(self, method):
        unit_msg = self._unit_msg()

        forward_msgs = self._forward_pass(method)
        backward_msgs = self._backward_pass(method)

        beliefs = {}
        for i in range(1, self._len + 1):

            y_i = self._observed[i - 1]
            msg_left = forward_msgs.get((i - 1 , i), unit_msg)
            msg_right = backward_msgs.get((i + 1 , i), unit_msg)

            belief_func = lambda x_i: self._phi(x_i, y_i) * msg_left[x_i] * msg_right[x_i]
            z_norm = sum([belief_func(v) for v in self._possible_values])

            beliefs[i] = [(1 / z_norm) * belief_func(x_i) for x_i in self._possible_values]

            if abs(beliefs[i][0] - 1 / len(self._possible_values)) < 0.001: ### DEBUG
                print("WARNING: May cause numerical issue for MAP!")

        return beliefs

    def _get_marginal_beliefs(self):
        return self._get_beliefs(method='sum_product')

    def _get_max_apostriori_beliefs(self):
        beliefs = self._get_beliefs(method='max_product')
        return [self._possible_values[np.argmax(b)] for k, b in beliefs.items()]
        

def f_phi(x, y_observed, b=[-0.32, 0.4]):
    return np.round(exp(b[y_observed] * (x - 0.5)), 2)
    
def f_psi(xi, xj, j=1):
    j = 1
    return np.round(exp(j * (xi - 0.5) * (xj - 0.5)), 2)