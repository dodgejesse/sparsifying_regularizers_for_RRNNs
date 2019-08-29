import torch

def identity(x):
    return x

def zero(data, *size):
    return data.new(*size).zero_()

def one(data, *size):
    return data.new(*size).zero_() + 1.

def neg_infinity(data, *size):
    return -100 * one(data, *size)

class Semiring:
    def __init__(self,
                 type,
                 zero,
                 one,
                 plus,
                 times,
                 conditional_times,
                 from_float,
                 to_float,
                 activation):
        self.type = type
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.conditional_times = conditional_times
        self.from_float = from_float
        self.to_float = to_float
        self.activation = activation


# element-wise plus, times
PlusTimesSemiring = \
    Semiring(
        0,
        zero,
        one,
        torch.add,
        torch.mul,
        torch.mul,
        identity,
        identity,
        identity
    )

# element-wise max, plus
MaxPlusSemiring = \
    Semiring(
        1,
        neg_infinity,
        zero,
        torch.max,
        torch.add,
        lambda x,y: x,
        identity,
        identity,
        torch.sigmoid
    )

# element-wise max, times. in log-space
MaxTimesSemiring = \
    Semiring(
        2,
        neg_infinity,
        one,
        torch.max,
        torch.mul,
        lambda x,y: x,
        identity,
        identity,
        torch.sigmoid
    )

def LogSum(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))

# element-wise max, times. in log-space
LogSemiring = \
    Semiring(
        3,
        neg_infinity,
        zero,
        # lambda x, y: torch.log(torch.exp(x) + torch.exp(y)),
        LogSum,
        torch.add,
        lambda x,y: x,
        identity,
        identity,
        torch.sigmoid
    )
