
PROPAGATION_METHODS =  ['sum_product', 'max_product']


class HMM(object):
    
    def __init__(*args, **kwargs):
        super(HMM).__init__(*args, **kwargs)
    
    def get_marginal_beliefs(self):
        return self._get_marginal_beliefs()
        
    def get_max_apostriori_beliefs(self):
        return self._get_max_apostriori_beliefs()
            
    def update_observed(self, observed):
        self._update_observed(observed)
    
    def _validate_propagation_method(self, method):
        if method not in PROPAGATION_METHODS:
            raise ValueError('method not in {}'.format(PROPAGATION_METHODS))

    def _normalize_msg(self, msg):
        """ 
        Normalize a message to avoid growing to infinity or decaying to zero.
        
        Params:
        --------
        msg: dict, the input message (for every possible believed value).
        """
        total = sum(msg.values())
        norm_msg = {k: v / total for k, v in msg.items()}
        return norm_msg