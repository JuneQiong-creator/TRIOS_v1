"""Fixed-representation evaluation; supplied full-data states are not retrained.
Applicable to original-five-component CCLE contract, not HB native domain LOO.
"""
import numpy as np
from .ccle_classifier import classify
from .evaluation import project,task_components,require

def evaluate_fixed_scores(scores,ids,full_labels,tau1,tau2,pfit,algorithm,context):
    x=np.asarray(scores,float);ids=np.asarray(ids,str);labels=np.asarray(full_labels,str)
    require(len(x)==len(ids)==len(labels) and len(set(ids))==len(ids),'INPUT_KEYS')
    trios=algorithm=='AB'
    require(np.array_equal(project(x,tau1,tau2,trios),labels),'FROZEN_FULL_PROJECTION')
    folds=[];R=float(np.ptp(x))
    for i,pid in enumerate(ids):
        keep=np.arange(len(x))!=i
        f=classify(x[keep],ids[keep],algorithm,context+'|PERTURB|'+pid)
        q=dict(attempted=True,success=bool(f.get('valid')))
        if q['success']:
            predicted=project(x,f['tau1'],f['tau2'],trios)
            q.update(d_tau_squared=.5*(((f['tau1']-tau1)/R)**2+((f['tau2']-tau2)/R)**2),
                     omitted_match=float(predicted[i]==labels[i]),
                     retained_match_fraction=float(np.mean(predicted[keep]==labels[keep])))
        else:q['failure_code']=f.get('failure_code','CLASSIFIER_FAILURE')
        folds.append(q)
    return task_components(x,labels,tau1,tau2,pfit,folds),folds
