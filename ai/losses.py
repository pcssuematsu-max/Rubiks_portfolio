"""Local loss functions and probability helpers for Rubiks AI."""

import numpy as np

def sigmoid(x):
    clipped_x = np.clip(x,-60.0,60.0)
    return 1.0 / (1.0 + np.exp(-clipped_x))

class Q_loss:
    def __init__(self,g):
        self.args = np.zeros(0,dtype = 'i')
        self.x = np.zeros(0,dtype = 'f')
        self.rewards = np.zeros(0,dtype = 'f')
        self.maxargs = np.zeros(0,dtype = 'i')
        self.v = np.zeros(0,dtype = 'f')
        self.X = np.zeros(0,dtype = 'i')
        self.g = g

        

    def forward(self,x,args,rewards):
        self.args = args
        self.x = x
        self.rewards = rewards

        self.maxargs = np.argmax(x[:,1:],axis = 0)

        self.X = np.arange(1,x.shape[1])

        self.v = self.x[self.args,self.X - 1] - self.rewards - self.g * self.x[self.maxargs,self.X]


        return np.sum(self.v ** 2)/2
        #return np.sum(huber(self.v))

    def backward(self):
        dO = np.zeros_like(self.x)

        #dv = self.v.copy()
        #dv[dv >= 1] = 1
        #dv[dv <= -1] = -1
        #dO[self.args,self.X - 1] += dv
        
        dO[self.args,self.X - 1] += self.v
        dO[self.maxargs,self.X] -= self.g * self.v
        


        return dO
        
class MSE:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')

    def forward(self,x,t,w = None):
        self.x = x
        self.t = t
        if w is None:
            self.w = np.ones_like(t,dtype = 'f')
        else:
            self.w = w

        return np.sum(self.w * (x - t) ** 2) / 2
    
    def backward(self):
        return self.w * (self.x - self.t)


class BCEWithLogits:
    """Binary cross entropy that takes raw logits and avoids sigmoid saturation in backprop."""

    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')

    def forward(self,x,t,w = None):
        self.x = x
        self.t = t
        if w is None:
            self.w = np.ones_like(t,dtype = 'f')
        else:
            self.w = w
        clipped_x = np.clip(x,-60.0,60.0)
        self.y = 1.0 / (1.0 + np.exp(-clipped_x))
        loss = np.maximum(x,0.0) - x * t + np.log1p(np.exp(-np.abs(x)))
        return np.sum(self.w * loss)

    def backward(self):
        return self.w * (self.y - self.t)
	    

class Soft_Target_Cross_Entropy:
    def __init__(self):
        self.y = np.zeros((0,0),dtype = 'f')
        self.t = np.zeros((0,0),dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')

    def forward(self,x,t,w = None):
        self.t = t
        self.y = softmax(x)
        if w is None:
            self.w = np.ones((1,t.shape[1]),dtype = 'f')
        else:
            self.w = w
        return -np.sum(self.w * t * np.log(self.y + 1.0e-7))

    def backward(self):
        return self.w * (self.y - self.t)


def softmax(x):
    y = x.copy()
    y -= np.max(y,axis = 0)
    y = np.exp(y)
    

    return y / np.sum(y,axis = 0)


def softmax_H(x):
    y = x.copy()
    y -= np.max(y,axis = 1)
    y = np.exp(y)
    

    return y / np.sum(y,axis = 1)    




def cross_entropy(y,t):
    return -np.sum(t * np.where(y != 0 , np.log(y),-100))



        

class Softmax_Cross_Entropy:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.Indices = None

    def forward(self,x,args,rewards,Indices,t = None):
        self.Indices = Indices
        self.x = x
        self.y = np.zeros((self.x.shape[0],0),dtype = 'i')
        self.t = np.zeros((self.x.shape[0],0),dtype = 'i')
        for i in range(len(Indices) - 1):
            self.y = np.c_[self.y,softmax(x[:,Indices[i]:Indices[i+1]-1])]

        if t is None:
            self.t = np.zeros_like(self.y,dtype = 'f')
            self.t[args,np.arange(self.t.shape[1])] = 1
        else:
            self.t = t
        return cross_entropy(self.y,self.t)

    def backward(self):
        dO = np.zeros_like(self.x)
        for i in range(len(self.Indices) -1):
            dO[:,self.Indices[i]:self.Indices[i+1] - 1] = self.y[:,self.Indices[i] - i:self.Indices[i+1] - i - 1] - self.t[:,self.Indices[i] - i:self.Indices[i+1] - i - 1]
        return dO

class Myloss:
    def __init__(self, margin = 0.0):
        self.margin = float(margin)
        self.x = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.w = np.ones((1,0),dtype = 'f')
        self.Indices = None

    def forward(self,x,Indices,w = None):

        self.y = np.zeros((1,Indices[-1]),dtype = 'f')
        self.t = np.zeros((1,Indices[-1]),dtype = 'f')
        self.w = np.ones_like(self.y)
        state_weights = None if w is None else np.asarray(w,dtype = 'f').reshape(1,-1)
        if state_weights is not None and state_weights.shape[1] != self.y.shape[1]:
            raise ValueError('rank weights must match the state count')
        for i in range(len(Indices) - 1):
            start,end = Indices[i],Indices[i+1]
            if state_weights is not None:
                self.w[:,start:end] = state_weights[:,start:start + 1]
            logits = x[:,start:end].copy()
            # Compare every preceding state with the sequence endpoint.
            # Shifting only nonterminal logits makes a margin of zero exactly
            # equivalent to the original endpoint-softmax objective.
            logits[:,:-1] += self.margin
            self.y[:,start:end] = softmax_H(logits)
            self.t[0,Indices[i+1]-1] = 1.0
        
        return -np.sum(self.w * self.t * np.where(self.y != 0,np.log(self.y),-100))
        
        #self.x = x
        #self.Indices = Indices
        #self.y = np.zeros((1,0),dtype = 'f')
        #for i in range(len(Indices) - 1):
        #    self.y = np.c_[self.y,self.x[:,Indices[i] + 1:Indices[i + 1]] - self.x[:,Indices[i]:Indices[i + 1] - 1]]
        
        #self.y = sigmoid(self.y)

        #return np.sum(self.y)
        

    def backward(self):
        return self.w * (self.y - self.t)
        #dO = np.zeros_like(self.x)
        #for i in range(len(self.Indices) - 1):
        #    dO[:,self.Indices[i]:self.Indices[i + 1] - 1] += self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)] * (1.0 - self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)])
        #    dO[:,self.Indices[i] + 1:self.Indices[i + 1]] -= self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)] * (1.0 - self.y[:,self.Indices[i] - i:self.Indices[i + 1] - (i + 1)])  
                        
        #return dO


class MyLoss2:
    def __init__(self):
        self.x = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.Indices = None


    def forward(self,x,Indices,steps_to_goal = None,gamma = None):
        self.x = x
        self.Indices = Indices
        if gamma is None:
            gamma = 1.0
        if steps_to_goal is None:
            steps_to_goal = self._default_steps_to_goal(Indices)
        dtype = self.x.dtype
        steps_to_goal = np.asarray(steps_to_goal,dtype = dtype).reshape(-1)
        self.y = np.zeros_like(self.x,dtype = dtype)
        self.t = np.zeros_like(self.x,dtype = dtype)
        for i in range(len(Indices) - 1):
            start = Indices[i]
            end = Indices[i + 1]
            if end <= start:
                continue
            self.y[:,start:end] = softmax_H(self.x[:,start:end])
            target = np.asarray(gamma,dtype = dtype) ** steps_to_goal[start:end]
            target_sum = np.sum(target)
            if target_sum <= 0:
                target = np.ones_like(target,dtype = dtype) / max(1,len(target))
            else:
                target = target / target_sum
            self.t[:,start:end] = target.reshape(1,-1)
        return cross_entropy(self.y,self.t)
    

    def backward(self):
        return self.y - self.t

    def _default_steps_to_goal(self, Indices):
        steps = []
        for i in range(len(Indices) - 1):
            start = Indices[i]
            end = Indices[i + 1]
            steps.extend(range(end - start - 1,-1,-1))
        return steps


class MyLoss2Pairwise:
    """Adjacent softplus ranking loss for the steps_to_goal auxiliary term."""

    def __init__(self, margin = 0.2, reduction = 'sum'):
        self.margin = float(margin)
        if reduction not in ('sum', 'trajectory_mean'):
            raise ValueError(f'unknown pairwise reduction: {reduction}')
        self.reduction = reduction
        self.x = np.zeros(0,dtype = 'f')
        self.diffs = np.zeros(0,dtype = 'f')
        self.y = np.zeros(0,dtype = 'f')
        self.Indices = None
        self.pair_scales = np.zeros((1,0),dtype = 'f')

    def forward(self,x,Indices,w = None):
        self.x = x
        self.Indices = Indices
        self.diffs = np.zeros((1,0),dtype = x.dtype)
        weights = None if w is None else np.asarray(w,dtype = x.dtype).reshape(-1)
        if weights is not None and weights.size != x.shape[1]:
            raise ValueError('pairwise weights must match the state count')
        scales = []
        for i in range(len(Indices) - 1):
            start = Indices[i]
            end = Indices[i + 1]
            if end - start <= 1:
                continue
            self.diffs = np.c_[self.diffs,self.x[:,start + 1:end] - self.x[:,start:end - 1]]
            scale = 1.0 / (end - start - 1) if self.reduction == 'trajectory_mean' else 1.0
            if weights is None:
                scales.extend([scale] * (end - start - 1))
            else:
                scales.extend(scale * weights[start:end - 1])
        self.pair_scales = np.asarray(scales,dtype = x.dtype).reshape(1,-1)
        self.y = sigmoid(self.diffs - self.margin)
        return np.sum(self.pair_scales * np.logaddexp(0.0,self.margin - self.diffs))

    def backward(self):
        dO = np.zeros_like(self.x)
        if self.Indices is None:
            return dO
        D = (self.y - 1.0) * self.pair_scales
        diff_index = 0
        for i in range(len(self.Indices) - 1):
            start = self.Indices[i]
            end = self.Indices[i + 1]
            count = end - start - 1
            if count <= 0:
                continue
            d = D[:,diff_index:diff_index + count]
            dO[:,start:end - 1] -= d
            dO[:,start + 1:end] += d
            diff_index += count
        return dO


def adjacent_mirror_pairs(start, end):
    """Return each adjacent and nonduplicate mirrored route pair in O(length)."""
    count = end - start
    if count < 2:
        empty = np.zeros(0,dtype='i')
        return empty,empty
    left = np.arange(start,end - 1,dtype='i')
    right = left + 1
    mirror_offsets = np.arange(count // 2,dtype='i')
    mirror_left = start + mirror_offsets
    mirror_right = end - 1 - mirror_offsets
    nonadjacent = mirror_right - mirror_left > 1
    if np.any(nonadjacent):
        left = np.concatenate((left,mirror_left[nonadjacent]))
        right = np.concatenate((right,mirror_right[nonadjacent]))
    return left,right


class RoutePairHuber:
    """Huber regression on adjacent and mirrored route Value differences."""

    def __init__(self, margin=1.0, delta=1.0):
        self.margin = float(margin)
        self.delta = max(1.0e-8,float(delta))
        self.gradient = np.zeros((1,0),dtype='f')
        self.pair_count = 0
        self.pair_weight_sum = 0.0
        self.pair_abs_error_sum = 0.0

    def forward(self, x, indices, w=None):
        self.gradient = np.zeros_like(x)
        self.pair_count = 0
        self.pair_weight_sum = 0.0
        self.pair_abs_error_sum = 0.0
        weights = None if w is None else np.asarray(w,dtype=x.dtype).reshape(-1)
        if weights is not None and weights.size != x.shape[1]:
            raise ValueError('pairwise weights must match the state count')
        loss = 0.0
        for start,end in zip(indices[:-1],indices[1:]):
            left,right = adjacent_mirror_pairs(start,end)
            if left.size == 0:
                continue
            # Adding mirrored pairs should not silently increase a long
            # route's total weight relative to the old adjacent-only loss.
            scale = (end - start - 1) / left.size
            pair_weights = np.ones(left.size,dtype=x.dtype) if weights is None else weights[left]
            pair_weights = scale * pair_weights
            errors = x[0,right] - x[0,left] - self.margin * (right - left)
            absolute = np.abs(errors)
            quadratic = np.minimum(absolute,self.delta)
            loss += float(np.sum(pair_weights * (
                0.5 * quadratic ** 2 + self.delta * (absolute - quadratic)
            )))
            derivatives = pair_weights * np.clip(errors,-self.delta,self.delta)
            np.add.at(self.gradient[0],left,-derivatives)
            np.add.at(self.gradient[0],right,derivatives)
            self.pair_count += int(left.size)
            self.pair_weight_sum += float(np.sum(pair_weights))
            self.pair_abs_error_sum += float(np.sum(pair_weights * absolute))
        return loss

    def backward(self):
        return self.gradient


class Huber:
    """Smooth L1 regression for calibrated Search2 step-distance values."""

    def __init__(self, delta = 1.0, reduction = 'sum'):
        self.delta = max(1.0e-8,float(delta))
        if reduction not in ('sum','mean','trajectory_sum'):
            raise ValueError(f'unknown Huber reduction: {reduction}')
        self.reduction = reduction
        self.x = np.zeros(0,dtype = 'f')
        self.t = np.zeros(0,dtype = 'f')
        self.diff = np.zeros(0,dtype = 'f')
        self.trajectory_scales = np.zeros(0,dtype = 'f')

    def forward(self,x,t,indices = None):
        self.x = x
        self.t = np.asarray(t,dtype = x.dtype).reshape(x.shape)
        self.diff = self.x - self.t
        absolute = np.abs(self.diff)
        quadratic = np.minimum(absolute,self.delta)
        linear = absolute - quadratic
        element_loss = 0.5 * quadratic ** 2 + self.delta * linear
        self.trajectory_scales = np.ones_like(self.diff,dtype = self.diff.dtype)
        if self.reduction == 'trajectory_sum':
            total = 0.0
            if indices is None:
                indices = (0,self.diff.shape[1])
            for index in range(len(indices) - 1):
                start,end = indices[index],indices[index + 1]
                count = max(0,end - start)
                if count:
                    total += np.sum(element_loss[:,start:end]) / count
                    self.trajectory_scales[:,start:end] = 1.0 / count
            return total
        total = np.sum(element_loss)
        if self.reduction == 'mean' and self.diff.size:
            return total / self.diff.size
        return total

    def backward(self):
        gradient = np.where(
            np.abs(self.diff) <= self.delta,
            self.diff,
            self.delta * np.sign(self.diff),
        )
        if self.reduction == 'mean' and self.diff.size:
            gradient = gradient / self.diff.size
        elif self.reduction == 'trajectory_sum':
            gradient = gradient * self.trajectory_scales
        return gradient
