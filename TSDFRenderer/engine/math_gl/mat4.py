from __future__ import print_function

from OpenGL.GL import *
from ctypes import *
from engine.math_gl.vec3 import *
from engine.math_gl.vec4 import *

import sys
import math
import copy

# [Xx Yx Zx Tx]  [0 4 8  12]
# [Xy Yy Zy Ty]  [1 5 9  13]
# [Xz Yz Zz Tz]  [2 6 10 14]
# [0  0  0  1 ]  [3 7 11 15]

class mat4(object):
    def __init__(self,*data):
        # ctype array to make it directly callable by tutorials.
        self._data = (GLfloat * 16)()

        # special case, empty constructor fills matrix w/ zeroes
        if len(data) == 0:
            self.data = self.zeroes().data
        else:
            self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,data):
         # mat4.data(0,1,2,.....,15)
        if len(data) == 16:
            for i,d in enumerate(data): self._data[i] = data[i]

        # mat4.data([0,1,2,.....,15])
        if len(data) == 1:
            for i,d in enumerate(data[0]): self._data[i] = data[0][i]

    def copy(self):
        """ Create a new copy of matrix
        """
        return copy.deepcopy(self)

    def __getitem__(self,row):
        """ Allow matrix indexing using [row][col] notation.
            Equivalent to C++ operator[](int row, int col)
        """
        return pointer(GLfloat.from_address(addressof(self._data) + sizeof(GLfloat) * row * 4))

    @staticmethod
    def zeroes():
        """ Fill w/ zeroes
        """
        return mat4.fill(0)

    @staticmethod
    def fill(v):
        return mat4([v for i in range(0,16)])

    @staticmethod
    def identity():
        return mat4(
            1,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,1)

    @staticmethod
    def perspective(fov_deg,aspect,z_near,z_far):
        assert(aspect != 0.0)
        assert(z_near != z_far)

        fov = math.radians(fov_deg)
        tan_half_fov = math.tan(fov / 2.0)

        m = mat4.zeroes()
        m[0][0] = 1.0 / (aspect * tan_half_fov)
        m[1][1] = 1.0 / (tan_half_fov)
        m[2][2] = -(z_far+z_near) / (z_far - z_near)
        m[2][3] = -1.0
        m[3][2] = -(2.0 * z_far * z_near) / (z_far - z_near)
        return m

    @staticmethod
    def lookat(eye,center,up):
        f = (center-eye).normalized()
        s = (vec3.cross(f,up).normalized())
        u = (vec3.cross(s,f))

        m = mat4.identity()
        m[0][0] = s.x
        m[1][0] = s.y
        m[2][0] = s.z

        m[0][1] = u.x;
        m[1][1] = u.y;
        m[2][1] = u.z;

        m[0][2] =-f.x;
        m[1][2] =-f.y;
        m[2][2] =-f.z;

        m[3][0] =-vec3.dot(s, eye);
        m[3][1] =-vec3.dot(u, eye);
        m[3][2] = vec3.dot(f, eye);
        return m

    def transpose(self):
        new_mat = mat4()
        for r in range(0,4):
            for c in range(0,4):
                new_mat[c][r] = self[r][c]

    def transposed(self):
        new_mat = self.transpose()
        self._data = new_mat._data

    def rotatex(self,angle):
        rad = math.radians(angle)
        m = self
        m[0][2] = math.cos(rad)
        m[0][3] = math.sin(rad)
        m[2][0] = -math.sin(rad)
        m[2][2] = math.cos(rad)

    def translate(self,vec3):
        self._data[12] = vec3.x
        self._data[13] = vec3.y
        self._data[14] = vec3.z

    def __mul__(self,other):
        m = mat4.zeroes()

        # swap matrix multiplication order to account for right sided (column oriented) multiplcation
        # glm was the basis for this code. (their code is much prettier)
        a = other#self
        b = self#other

        for r in range(0,4):
            for c in range(0,4):
                for i in range(0,4):
                    m[r][c] += a[r][i] * b[i][c]

        #print("result--\n",m)
        return m

    @staticmethod
    def arith(op,a,b):
        """ Perform arithmetic `op` on `a` and `b'
        """
        rtype = type(b)
        if rtype is mat4:
            ret = mat4()
            for r in range(0,4):
                for c in range(0,4):
                    ret[r][c] = op(a[r][c],b[r][c])
            return ret
        elif rtype is float or rtype is int:
            raise NotImplementedError("rtype vec4 not yet supported, but it should be. ")

    @staticmethod
    def arith_inline(op,a,b):
        """ Perform arithmetic `op` on `self` and `b'
        """
        rtype = type(b)
        if rtype is mat4:
            for r in range(0,4):
                for c in range(0,4):
                    a[r][c] = op(a[r][c],b[r][c])
            return a

    def __add__(self, other):return mat4.arith(operator.add,self,other)
    def __iadd__(self,other):return mat4.arith_inline(operator.add,self,other)
    def __radd__(self,other):return mat4.arith(operator.add,self,other)

    def __sub__(self, other):return mat4.arith(operator.sub,self,other)
    def __isub__(self,other):return mat4.arith_inline(operator.sub,self,other)
    def __rsub__(self,other):return mat4.arith(operator.sub,self,other)

    def __eq__(self,other):
        for i in range(0,16):
            if math.fabs(self.data[i] - other.data[i]) >= sys.float_info.epsilon:return False
        return True

    def __ne__(self,other):
        return not (self==other)

    def __str__(self):
        return "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n"%(
            self._data[0],self._data[1],self._data[2],self._data[3],
            self._data[4],self._data[5],self._data[6],self._data[7],
            self._data[8],self._data[9],self._data[10],self._data[11],
            self._data[12],self._data[13],self._data[14],self._data[15])

    def __unicode__(self):
        print("unicode");
        return "%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n"%(
            self._data[0],self._data[1],self._data[2],self._data[3],
            self._data[4],self._data[5],self._data[6],self._data[7],
            self._data[8],self._data[9],self._data[10],self._data[11],
            self._data[12],self._data[13],self._data[14],self._data[15]
        )