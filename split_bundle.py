import numpy as np
import time
import sympy

class line_bundle():
    '''
    line bundle on P1 is an integer
    '''
    def __init__(self,type):
        self._type = type
        self.z0 = sympy.symbols('z_0')
        self.z1 = sympy.symbols('z_1')
    def rank(self):
        return 1
    def twist(self,n):
        self._type += n
    def h0(self):
        if self._type>=0:
            return self._type+1
        else:
            return 0
    def h1(self):
        if self._type<0:
            return self._type+1
        else:
            return 0
    def show_vars(self):
        print(self.z0,self.z1)


class split_bundle:
    '''
    A split bundle object on P1 is an np array. Further
    functionality such as twisting, h0, h1 and basis for H0
    and H1 are also provided
    '''

    def __init__(self,split_type):
        self._split_type = np.array(split_type)
    def rank(self):
        return len(self._split_type)
    def update_split_type(self,twist):
        self._split_type += twist

# run some tests
if __name__ == '__main__':
    print('start of tests...')
    time.sleep(1)
    L = line_bundle(5)
    L.show_vars()
