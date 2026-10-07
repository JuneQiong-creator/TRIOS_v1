import warnings
import numpy as np
from scipy.integrate import IntegrationWarning
from .evaluation import require,project
from . import classifiers_v11_frozen as classifiers
from .ab_ccle import fit_ms
native={"fit_ms":fit_ms}

def classify(x,ids,alg,context):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r=native['fit_ms'](x,ids) if alg=='AB' else classifiers.fit_classifier(x,ids,alg,context)[0]
    bad=[str(w.message) for w in caught if issubclass(w.category,(RuntimeWarning,IntegrationWarning))]
    require(not bad,'NUMERICAL_CLASSIFIER_WARNING:'+repr(bad))
    if r.get('valid'):
        labels=np.asarray(r['labels']);require(np.isfinite([r['tau1'],r['tau2']]).all(),'VALID_CUTOFF_NONFINITE')
        require(np.array_equal(labels,project(x,r['tau1'],r['tau2'],alg=='AB')),'VALID_PROJECTION_MISMATCH')
        require(all(np.sum(labels==v)>0 for v in 'LIH'),'VALID_EMPTY_GROUP')
    return r
