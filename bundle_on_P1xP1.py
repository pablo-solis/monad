import numpy as np
import sympy
from discrete_bundle import base_discrete_bundle, bundle_on_P1
from sympy.physics.quantum import TensorProduct #for H0 method
import time # only for tests

class split_bundle_P1xP1(base_discrete_bundle):
    '''
    A split bundle object on P1xP1 is an np array.
    each element is a pair (a,b)

    Further
    functionality such as twisting, h0, h1,h2 and basis for H0 are also provided
    '''
    def __init__(self,*args, splitting = None,vars = ['z_0','z_1','w_0','w_1']):
        # allow for different inits for rank 1 cases
        if splitting == None and len(args) == 2:
            # split_bundle_P1xP1(a,b)
            self._splitting = list(args)
            self._rank = 1
        elif splitting == None and len(args) ==1:
            # split_bundle_P1xP1([a,b])
            self._splitting = list(args)[0]
            self._rank = 1
        elif splitting == None:
            raise SyntaxError('*args can have at most 2 inputs')
        elif len(splitting) == 2 and isinstance(splitting[0],int):
            # the case split_bundle_P1xP1(splitting = [a,b])
            self._splitting = splitting
            self._rank = 1 #<-- key distinction
        elif len(splitting) == 1:
            # the case split_bundle_P1xP1(splitting = [[a,b]])
            self._splitting = splitting[0]
            self._rank =1
        else:
            # higher rank case
            self._splitting = splitting
            self._rank = len(splitting)

        # set vars used for H0 method
        self.z0 = vars[0]
        self.z1 = vars[1]
        self.w0 = vars[2]
        self.w1 = vars[3]

    def rank(self):
        # various super class methods have dependency on rank()
        return self._rank

    def __str__(self):
        if self._rank==1:
            return 'O('+str(self._splitting[0])+','+str(self._splitting[1])+')'
        else:
            lst = ['O('+str(n[0])+','+str(n[1])+')++' for n in self._splitting]
            string = ''.join(lst)
            return string[:-2]
    def kunneth_0(self,ab):
        # ab should be a list of len 2
        a = ab[0]
        b = ab[1]
        return self.h0_from_chi(a)*self.h0_from_chi(b)
    def kunneth_1(self,ab):
        # ab should be a list of len 2
        a = ab[0]
        b = ab[1]
        return self.h0_from_chi(a)*self.h1_from_chi(b)+self.h1_from_chi(a)*self.h0_from_chi(b)
    def kunneth_2(self,ab):
        # ab should be a list of len 2
        a = ab[0]
        b = ab[1]
        return self.h1_from_chi(a)*self.h1_from_chi(b)

    def h0(self):
        return self.additive_func(self.kunneth_0)
    def h1(self):
        return self.additive_func(self.kunneth_1)
    def h2(self):
        return self.additive_func(self.kunneth_2)

    def H0(self):
        '''
        This method is built from the H0 method for bundle_on_P1
        but it is not inheritted from it
        '''

        if self.h0()>0 and self._rank == 1:
            # rank 1 case first
            a = self._splitting[0]
            b = self._splitting[1]

            A = bundle_on_P1(a, vars = [self.z0,self.z1])
            B = bundle_on_P1(b,vars = [self.w0,self.w1])

            # list comprehension works from the outside in
            return [TensorProduct(i,j) for i in A.H0() for j in B.H0()]
        else:
            return 'Warning, higher rank not implemented'


    def __add__(self,other):
        # use list() to get correct behavior with +
        # in case self._splitting was initialized with np array or
        # other iterable
        if self._rank==1 and other._rank==1:
            return split_bundle_P1xP1(splitting = [self._splitting,other._splitting])
        elif self._rank ==1:
            return split_bundle_P1xP1(splitting = [self._splitting]+list(other._splitting))
        elif other._rank==1:
            return other+self
        else:
            return split_bundle_P1xP1(splitting = list(self._splitting)+list(other._splitting))
    def __mul__(self,other):
        # use np arrays to get desired behavior
        if self._rank==1 and other._rank==1:
            type = np.array(self._splitting)+np.array(other._splitting)
            # convert to list of ints so class understands __init__
            type = [int(i) for i in type]
        elif self._rank == 1 or other._rank ==1:
            # figure out which one is rank 1
            single = self if self._rank==1  else other
            multiple = self if self._rank>1 else other

            type = [np.array(single._splitting)+np.array(l) for l in multiple._splitting]
            type = [list(l) for l in type]
        else:
            # genereal cases
            type = [ np.array(self_i)+np.array(other_j) for self_i in self._splitting for other_j in self._splitting]
            type = [list(i) for i in type]
        return split_bundle_P1xP1(splitting = type)
        # fix this...



if __name__=='__main__':
    print('start test')
    E = split_bundle_P1xP1(1,2)
    print('E = split_bundle_P1xP1(1,2)',E)
    print(type(E._splitting))
    print(len(E._splitting))
    print(E._splitting)

    F = split_bundle_P1xP1(splitting = [3,2])
    print('F = split_bundle_P1xP1([3,2])',F)

    G = split_bundle_P1xP1(splitting = [[2,3],[4,5]])
    print('split_bundle_P1xP1(splitting = [[2,3],[4,5]])',G)

    print('cohomology of E')
    print(E.h0())
    print(E.h1())
    print(E.h2())

    time.sleep(1)
    print('add bundles E+F')
    print(E+F)
    print('add E+G')
    print(E+G)
    time.sleep(1)
    print('try product')
    print('E*F',(E*F))

    print('(E+G)*F',(E+G)*F)

    time.sleep(1)
    print('test H0')
    print(F)
    sympy.pprint(F.H0())
