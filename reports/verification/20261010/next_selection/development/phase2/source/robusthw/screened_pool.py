"""Two fixed data-informed candidates; the remaining pool/ranking is v3."""
import numpy as np
from .history import proposal_pool as original_pool

def proposal_pool(problem, history, incumbent, *, count, rng, guard=None, remaining=None):
    means=[q @ x for q,x in zip(problem.weights,problem.samples)]
    scales=[np.sqrt(q @ ((x-mean)**2))
            for q,x,mean in zip(problem.weights,problem.samples,means)]
    difference=np.abs(means[0]-means[1])
    denominator=scales[0]+scales[1]+1e-12
    ordinary=difference/denominator
    robust=np.maximum(0.,difference-problem.theta.sum())/denominator
    mandatory=[]
    for values in (ordinary,robust):
        support=tuple(sorted(np.argsort(-values,kind="stable")[:problem.budget]))
        if support not in mandatory:
            mandatory.append(support)
    original=original_pool(problem,history,incumbent,count=count,rng=rng,
                           guard=guard,remaining=remaining)
    supports=mandatory+[problem.support(z) for z in original
                         if problem.support(z) not in mandatory]
    return [problem.mask(s).astype(float) for s in supports[:count]]
