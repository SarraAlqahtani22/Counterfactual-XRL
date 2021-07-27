
import numpy as np
import scipy.stats
from scipy.special import kl_div

# calculate the kl divergence
def kl_divergence(p, q):
    epsilon = 0.0000001
    p = p + epsilon
    q = q + epsilon
    ans = sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))
    return ans



"""
# scipy.stats.entropy(x, y) or scipy.stats.entropy(px, py)
px = np.array([0.1, 0.5, 0.4])
py = np.array([0.001, 0.9, 0.099])
KL = scipy.stats.entropy(px, py) 
KLL = kl_divergence(px,py)
KLLL = sum(kl_div(px,py))
print(KL)
print(KLL)
print(KLLL)
"""
for i in range(100):
    a = np.random.random()
    print(a)
 

