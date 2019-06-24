import numpy as np
import sympy
import time #only for tests


class base_discrete_bundle:
    '''
    Abstract base class for line bundles on P1 and P1 x P1
    NOTE: rank,h0,h1 must be implemented by subclass
    '''
    def __init__(self,splitting):
        if isinstance(splitting,int):
            self._splitting = splitting
        else:
            self._splitting = sorted(splitting)

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
        raise NotImplementedError('Subclass of discrete_bundle must implement rank method.')


    def h0(self):
        '''definition will vary, must be implemented on subclass'''
        raise NotImplementedError('Subclass of discrete_bundle must implement h0 method.')

    def h1(self):
        '''definition will vary, must be implemented on subclass'''
        raise NotImplementedError('Subclass of discrete_bundle must implement h1 method.')

    def twist(self,tw):
        '''definition will vary, must be implemented on subclass'''
        raise NotImplementedError('Subclass of discrete_bundle must implement twist method.')


    def val_to_basis(self,val=1,i=0,rk=1):
        # create np array size rk with val in position i
        # uses zero indexing
        temp = [0]*rk
        temp[i] = val
        return temp

    def list_to_basis(self,lst,i=0,rk=1):
        n = len(lst)
        # create n vector of size rk where elements of lst appear
        # in ith position
        vecs = [ self.val_to_basis(a,i,rk) for a in lst]
        return vecs

    def one_element_arr(self,a):
        #this is when you don't know if a is a list or number but you wanted to feed it into something that expects a list
        if isinstance(a,(int,float)):
            return [a]
        else:
            return a

class bundle_on_P1(base_discrete_bundle):
    '''
    class to describe vector bundles on P1
    a bundle will generally be an np.array of ints
    except in the case of rank 1
    '''
    def __init__(self,splitting, vars = ['z_0','z_1']):
        super().__init__(splitting)
        # handle rank 1 case seperately
        if isinstance(splitting,int):
            self._rank = 1
        else:
            self._rank = len(self._splitting)
        # z0,z1 used to express basis for H0
        self.z0 = sympy.symbols(vars[0])
        self.z1 = sympy.symbols(vars[1])
    def rank(self):
        return self._rank

    # use str method to give str representation of a vector bundle
    def __str__(self):
        if self.rank()==1:
            return 'O('+str(self._splitting)+')'
        else:
            lst = ['O('+str(n)+')++' for n in self._splitting]
            string = ''.join(lst)
            return string[:-2]

    def h0(self):
        return self.additive_func(self.h0_from_chi)
    def h1(self):
        return self.additive_func(self.h1_from_chi)
    def H0(self):
        if self._rank==1 and self.h0()>=0:
            n = self._splitting
            return [sympy.Matrix([self.z0**i*self.z1**(n-i)]) for i in range(n+1)]
        else:
            rk = self._rank
            vecs = []
            for i in range(rk):
                type = self._splitting[i]
                lst = [self.z0**i*self.z1**(type-i) for i in range(type+1)]
                vecs += self.list_to_basis(lst,i,rk)
                vecs = [sympy.Matrix(a) for a in vecs]
            return vecs
    def __add__(self,other):
        # use list() to get correct behavior with +
        # in case self._splitting was initialized with np array or
        # other iterable
        if self._rank==1 and other._rank==1:
            return bundle_on_P1([self._splitting,other._splitting])
        elif self._rank ==1:
            return bundle_on_P1([self._splitting]+list(other._splitting))
        elif other._rank==1:
            return other+self
        else:
            return bundle_on_P1(list(self._splitting)+list(other._splitting))
    def __mul__(self,other):
        if self._rank==1 and other._rank==1:
            return bundle_on_P1(self._splitting*other._splitting)
        else:
            #use np.array to combine multiple cases into one

            self_arr = np.array(self.one_element_arr(self._splitting))
            other_arr = np.array(self.one_element_arr(other._splitting))
            #(n,1)*(1*m) = (n,m)
            self_resh = self_arr.reshape(len(self_arr),1)
            other_resh = other_arr.reshape(1,len(other_arr))
            product = self_resh*other_resh
            return bundle_on_P1(product.flatten())
    def twist(self,rk1):
        '''
        rk1 is either an integer or a rank 1 bundle
        '''
        if isinstance(rk1,int):
            other_bundle = bundle_on_P1(rk1)
            # this is how to update self
            self._splitting = (self*other_bundle)._splitting
            # note that self = self*other doesn't work
        elif isinstance(rk1,bundle_on_P1):
            self = self*other_bundle
        else:
            raise TypeError('can only twist by an integer or rank 1 bundle')


# run some tests
if __name__ == '__main__':
    print('start of tests...')
    time.sleep(1)
    E = bundle_on_P1(np.array([1,2]))
    # print(E)
    print('h^0(E) =',E.h0())
    sympy.pprint(E.H0())
    F = bundle_on_P1(5)
    time.sleep(0.9)
    print('F = ',F)
    print('H0 = ')
    # sympy.pprint(F.H0())
    print('F + F is ',F+F)
    print('E+F is', E+F)
    print('E+E', E+E)
    time.sleep(0.9)
    print('now test tensor product')
    print('F * F is ',F*F)
    print('E*F is', E*F)
    print('E*E', E*E)
    time.sleep(0.9)
    print('E*E has H0 basis:')
    sympy.pprint((E*E).H0())

    print(E)
    print('now twist by 2: E.twist(2)')
    E.twist(2)
    print(E)

    print(bundle_on_P1(np.array([2])))
