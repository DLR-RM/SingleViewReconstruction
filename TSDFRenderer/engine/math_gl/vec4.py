from __future__ import print_function

from OpenGL.GL import *

import sys
import math
import copy
import operator

class vec4(object):
    def __init__(self,x=0,y=0,z=0,w=0):
        self.data_ = (GLfloat * 4)()
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def copy(self):
        return copy.deepcopy(self)

    @property
    def x(self):
        return self.data_[0]
    @x.setter
    def x(self,value):
        self.data_[0] = value

    @property
    def y(self):
        return self.data_[1]
    @y.setter
    def y(self,value):
        self.data_[1] = value

    @property
    def z(self):
        return self.data_[2]
    @z.setter
    def z(self,value):
        self.data_[2] = value

    @property
    def w(self):
        return self.data_[3]
    @w.setter
    def w(self,value):
        self.data_[3] = value


    @staticmethod
    def arith(op,a,b):
        """ Perform arithmetic `op` on `a` and `b'
            Input:
                op(operator) - Python operator type
                a(vec4) - left hand side vector3 (
                    *type always equals vec4, support for int+vec4() is handled by _i<op>_ methods
                b(int,float,vec4) - right hand side int, float, or vector3.
            Notes:
                Python doesn't support method overloading in the C++ sense so this
                utility method performs an r-value type check.
            TODO:
                This method is not correct for vec4 multiplication or
                division.  It makes no sense to multiply vec4*vec4.
                Vec4 * scalar is fine.
        """
        print(op)
        rtype = type(b)
        if rtype is vec4:
            return vec4(op(a.x,b.x),op(a.y,b.y),op(a.z,b.z),op(a.w,b.w))
        elif rtype is float or rtype is int:
            return vec4(op(a.x,b),op(a.y,b),op(a.z,b),op(a.w,b))

    @staticmethod
    def arith_inline(op,a,b):
        """ Perform arithmetic `op` on `self` and `b'
            *See arith documentation for explanation.
            **arith_inline handles: my_vec4 += other_vec4 -or-
              my_vec4 += 3
        """
        rtype = type(b)
        if rtype is vec4:
            a.x = op(a.x,b.x)
            a.y = op(a.y,b.y)
            a.z = op(a.z,b.z)
            a.w = op(a.w,b.w)
            return a
        elif rtype is float or rtype is int:
            a.x = op(a.x,b)
            a.y = op(a.y,b)
            a.z = op(a.z,b)
            a.w = op(a.w,b)
            return a

    # todo: consider less visually awful approach to overloading.
    def __add__(self, other):return vec4.arith(operator.add,self,other)
    def __iadd__(self,other):return vec4.arith_inline(operator.add,self,other)
    def __radd__(self,other):return vec4.arith(operator.add,self,other)

    def __sub__(self, other):return vec4.arith(operator.sub,self,other)
    def __isub__(self,other):return vec4.arith_inline(operator.sub,self,other)
    def __rsub__(self,other):return vec4.arith(operator.sub,self,other)

    def __mul__(self, other):return vec4.arith(operator.mul,self,other)
    def __imul__(self,other):return vec4.arith_inline(operator.mul,self,other)
    def __rmul__(self,other):return vec4.arith(operator.mul,self,other)

    # for python 3
    def __truediv__(self, other):return vec4.arith(operator.truediv,self,other)
    def __itruediv__(self,other):return vec4.arith_inline(operator.truediv,self,other)
    def __rtruediv__(self,other):return vec4.arith(operator.truediv,self,other)

    def __div__(self, other):return vec4.arith(operator.div,self,other)
    def __idiv__(self,other):return vec4.arith_inline(operator.div,self,other)
    def __rdiv__(self,other):return vec4.arith(operator.div,self,other)


    def __eq__(self,other):
        if math.fabs(self.x - other.x) >= sys.float_info.epsilon:return False
        if math.fabs(self.y - other.y) >= sys.float_info.epsilon:return False
        if math.fabs(self.z - other.z) >= sys.float_info.epsilon:return False
        if math.fabs(self.w - other.w) >= sys.float_info.epsilon:return False
        return True

    def __ne__(self,other):
        return not (self==other)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return "%f %f %f %f"%(self.x,self.y,self.z,self.w)
