import numpy as np
import time
import sympy

class discrete_bundle:
    '''
    Abstract base class for line bundles on P1 and P1 x P1
    # NOTE: rank,h0,h1 must be implemented by subclass
    '''
    def __init__(self,splitting):
        self._splitting = splitting

    def twist(self,tw):
        self._splitting += tw

    def chi(self,n):
        '''
        cohomology calculations for P1 split into a positive
        or negative cases depending on whether n is >=-1 or <-2
        chi is always n+1
        '''
        return n+1
    def h0_from_chi(self,n):
        if self.chi(n)>=0:
            return self.chi(n)
        else:
            return 0
    def h1_from_chi(self,n):
        if self.chi(n)<0:
            return -self.chi(n)
        else:
            return 0

    # next we add capability to compute an additive function on
    # self._splitting
    # examples will be h0,h1 etc
    def additive_func(self,base_case):
        # base case is meant to handle the rank 1 cases
        # note also rank must be implemented in subclass
        if self.rank()==1:
            return base_case(self._splitting)
        else:
            # otherwise apply base_case to each element and sum
            return sum([base_case(n) for n in self._splitting])

    def rank(self):
        '''definition will vary, must be implemented on subclass'''
        raise NotImplementedError('Subclass of discrete_bundle must implement fit method.')


    def h0(self):
        '''definition will vary, must be implemented on subclass'''
        raise NotImplementedError('Subclass of discrete_bundle must implement fit method.')

    def h1(self):
        '''definition will vary, must be implemented on subclass'''
        raise NotImplementedError('Subclass of discrete_bundle must implement fit method.')


class bundle_on_P1(discrete_bundle):
    '''
    class to describe vector bundles on P1
    a bundle will generally be an np.array of ints
    except in the case of rank 1
    '''
    def __init__(self,splitting, vars = ['z_0','z_1']):
        # handle rank 1 case seperately
        if isinstance(splitting,int):
            self._splitting = splitting
        else:
            self._splitting = np.array(splitting)
        # z0,z1 used to express basis for H0
        self.z0 = sympy.symbols(vars[0])
        self.z1 = sympy.symbols(vars[1])
    def rank(self):
        if isinstance(self._splitting,int):
            return 1
        else:
            return len(self._splitting)

    # use str method to give str representation of a vector bundle
    def __str__(self):
        if self.rank()==1:
            return 'O('+str(self._splitting)+')'
        else:
            lst = ['O('+str(n)+')++' for n in self._splitting]
            string = ''.join(lst)
            return string[:-2]
        return 'O('+str(self._splitting)+')'
    def h0(self):
        return self.additive_func(self.h0_from_chi)
    def h1(self):
        return self.additive_func(self.h1_from_chi)
    def H0(self):
        if self.rank()==1 and self.h0()>=0:
            n = self._splitting
            return [self.z0**i*self.z1**(n-i) for i in range(n+1)]
        else:
            return 'higher rank not implemented yet'

class bundle_on_P1xP1:
    '''
    A split bundle object on P1 is an np array. Further
    functionality such as twisting, h0, h1 and basis for H0
    and H1 are also provided
    '''
# run some tests
if __name__ == '__main__':
    print('start of tests...')
    time.sleep(1)
    E = bundle_on_P1([2,3])
    print(E)
    print('h^0(E) =',E.h0())
    print(E.H0())
    F = bundle_on_P1(5)
    time.sleep(1)
    print('F = ',F)
    print('H0 = ')
    print(F.H0())
    print(np.array([1,2,3]).reshape(3,1))
