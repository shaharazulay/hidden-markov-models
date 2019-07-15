
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
            
class Memoize(object):
    
    def __init__(self, f):
        self.f = f
        self.memo = {}
        
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        
        return self.memo[args]