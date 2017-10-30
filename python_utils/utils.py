import numpy as np
from scipy.sparse.linalg import lsqr as sparse_lsqr
from scipy.optimize import nnls
from sklearn.utils import check_consistent_length, column_or_1d

def check_constraints_old(X, constraints):
    if not isinstance(constraints, dict):
        raise ValueError("constraints should be a dictionary")
    for k in ["sign", "value"]:
        if k not in constraints.keys():
            raise ValueError("attribute %s is missing in dictionary constraints" % k)
        check_consistent_length(X.T, constraints[k])
        column_or_1d(constraints[k])
        constraints[k] = constraints[k].ravel()
    if not all(x != 0 for x in constraints["sign"]):
        raise ValueError("constraints['sign'] cannot contain 0")
    constraints["sign"] = np.sign(constraints["sign"])
    return constraints

def check_constraints(X, constraints):
    check_consistent_length(X.T, constraints)
    column_or_1d(constraints)
    constraints = constraints.ravel()
    if not all(x != 0 for x in constraints):
        raise ValueError("constraints cannot contain 0")
    constraints = np.sign(constraints)
    return constraints

def optim_fun(X, y, is_constr):
    y = y.ravel()
    if is_constr:
        coef, residues = nnls(X, y)
    else:
        out = sparse_lsqr(X, y)
        coef = out[0]
        residues = out[3]
    return (coef, residues)
