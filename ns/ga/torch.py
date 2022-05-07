import numpy as np
import torch
import ns.lib.range

# This file is based on the module from PyGAD, with the exception that it supports chromosome "folding".  This allows more granularity for
# GA crossover: full/partial network modules can be crossed over individually instead of arbitrarily picking chromosome values.

def model_weights_as_vector(model, folds=None):
    '''
    Convert torch state dict to a flattened vector

    Parameters
    ----------
    model : torch.Module
      Model to convert
    folds : list of strings (optional)
      A list of module names that will be folded into a single "chromosome" for crossover.
      Partial starting match is okay, in that case multiple modules will be folded together.

    Returns
    -------
    weights_vector : np.ndarray
      Weights vector
    folds_out : list of ns.lib.range.DisjointRanges
      Ranges of indices in weights vector for each fold.
      Index ranges are inclusive on low end and exclusive on high end.
    '''

    vec_size = 0
    state_dict = model.state_dict()

    # Scan state dictionary to determine output vector length
    for weight in model.state_dict().values():
        vec_size += np.prod(weight.detach().shape)
    weights_vector = np.zeros(vec_size)

    # Create folds assignment list if we have folds
    folds_out = []
    if folds is not None:
        for i in range(len(folds)):
            folds_out.append(ns.lib.range.DisjointRanges())

    # Finally, populate vector and folds assignment
    cur_spot = 0
    for name, weights in state_dict.items():
        vector = weights.detach().cpu().numpy().flatten()
        n = len(vector)

        # Find fold and update index range
        if folds is not None:
            fld_idx = -1
            for i, fold_name in enumerate(folds):
                if name.startswith(fold_name):
                    fld_idx = i
                    break
            if fld_idx == -1:
                raise RuntimeError(f'Could not find fold for model component {name}.')

            folds_out[fld_idx].extend(ns.lib.range.Range(cur_spot, cur_spot + n))

        weights_vector[cur_spot:cur_spot+n] = vector
        cur_spot += n

    return weights_vector, folds_out


def model_weights_as_dict(model, weights_vector):
    '''
    Convert flattened vector of model weights back into torch state dict
    '''

    state_dict = model.state_dict()
    cur_spot = 0
    weights_vector = torch.Tensor(weights_vector).to(model.device)

    for key in state_dict.keys():
        weights = state_dict[key].detach()
        shape = weights.shape
        length = np.prod(weights.shape)

        state_dict[key] = weights_vector[cur_spot:cur_spot+length].reshape(shape)
        cur_spot += length

    return state_dict


class TorchGA:
    def __init__(self, model, num_solutions, model_fold_names=None, random_perturb=1.):
        self.model = model
        self.num_solutions = num_solutions
        self.population_weights = self.create_population(model_fold_names, random_perturb)

    def create_population(self, model_fold_names, random_perturb=1.):
        if model_fold_names is None:
            self.fold_names = list(self.model.state_dict().keys())
        else:
            self.fold_names = model_fold_names.copy()

        weights, self.folds = model_weights_as_vector(model=self.model, folds=self.fold_names)

        net_population_weights = []
        net_population_weights.append(weights)
        for idx in range(self.num_solutions-1):
            net_weights = weights + np.random.uniform(low=-random_perturb, high=random_perturb, size=weights.size)
            net_population_weights.append(net_weights)

        return net_population_weights
