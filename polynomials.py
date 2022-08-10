"""
Polynomials

from polynomials import Polynomial
p1 = Polynomial([2,0,3])
p2 = Polynomial([1,4,5])
p3 = p1 + p2
p4 = Polynomial([4,7,2,1])
p5 = p3 + p4
p6 = p4 + p3
p7 = p2 - p1
p8 = p4 - p1
p9 = p1 - p4

from polynomials import Polynomial
p10 = Polynomial([1,2,0,1])
p11 = Polynomial([0,0,1,1])
p12 = p10 * p11
p13 = Polynomial([1,0,1])
p14 = p11 * p13
p15 = p12 * p13

from polynomials import Polynomial
p16 = Polynomial([1,0,0,0,1])
p17 = Polynomial([1,0,1])
p18 = p16 // p17

from polynomials import Polynomial
p19 = Polynomial([1,1,1,1,1])
p20 = p19.derivative()

from polynomials import Polynomial
p21 = Polynomial([0,0,1])
p21.find_roots()

"""
from fractions import Fraction

class Polynomial:

    def __init__(self, entries):
        """
        The entries list for the polynomial starts at x^0 and ends at x^length-1 == x^degree
        The index in the list corresponds to the appropiate power of x
        """
        assert type(entries) is list
        self.ent = entries
        self.deg = len(entries)-1
        self.len = len(entries)
        self.roots = None
        self.derivative = None
        self.integral = None

    def get_coeff(self, i=None):
        if type(i) is int:
            return self.ent[i]
        else:
            return self.ent

    def update(self, i, val):
        if i == self.deg and val == 0:
            self.ent.pop()
            self.deg -= 1
            self.len -= 1
        else:
            self.ent[i] = val

    def calculus(self):
        self.derivative = self.derivative()
        self.integral = self.integral()


    #Polynomial operations
    def __add__(self, other):
        if type(other) is Polynomial:
            diff = self.deg - other.deg
            if diff < 0:
                lst = [x+y for (x,y) in zip(self.get_coeff(), other.get_coeff())] + other.get_coeff()[diff:]
                return Polynomial(lst)
            elif diff > 0:
                lst = [x+y for (x,y) in zip(self.get_coeff(), other.get_coeff())] + self.get_coeff()[-diff:]
                return Polynomial(lst)
            else:
                lst = [x+y for (x,y) in zip(self.get_coeff(), other.get_coeff())]
                if not all(map(lambda x: x==0, lst)):
                    while lst[-1] == 0:
                        lst.pop()
                return Polynomial(lst)
        elif type(other) is float or type(other) is int or type(other) is Fraction:
            return Polynomial([self.get_coeff(0)+other] + self.get_coeff()[1:])

    def __sub__(self, other):
        if type(other) is Polynomial:
            return self + Polynomial(list(map(lambda x: -x, other.get_coeff())))
        elif type(other) is float or type(other) is int or type(other) is Fraction:
            return Polynomial([self.get_coeff(0)-other] + self.get_coeff()[1:])

    def __mul__(self, other):
        def mult_helper(lst1, lst2):
            total = list(map(lambda y: lst1[0]*y, lst2))
            total = Polynomial(total)
            new_lst = lst1[1:]
            for i in range(len(new_lst)):
                add_zeros = [0 for _ in range(i+1)]
                placeholder = add_zeros + list(map(lambda y: new_lst[i]*y, lst2))
                total = total + Polynomial(placeholder)
            return total
        if type(other) is Polynomial:
            return mult_helper(self.get_coeff(), other.get_coeff())
        elif type(other) is float or type(other) is int or type(other) is Fraction:
            return Polynomial(list(map(lambda x: x * other, self.get_coeff())))

    def __truediv__(self, other):
        #Returns a polynomial division as a tuple containing the quotient and remainder
        assert self.deg >= other.deg

        def div_helper(lst1, lst2, poly):
            if len(lst1) < len(lst2):
                return (poly, Polynomial(lst1))
            diff = len(lst1) - len(lst2)
            zeros = [0 for _ in range(diff)]
            adjust = zeros + lst2
            num = adjust[-1] / lst1[-1]
            adjust = list(map(lambda x: num*x, adjust))
            p = Polynomial(lst1) - Polynomial(adjust)
            new_lst = zeros + [num]
            poly = poly + Polynomial(new_lst)
            return div_helper(p.get_coeff(), lst2, poly)

        return div_helper(self.get_coeff(), other.get_coeff(), Polynomial([0]))

    def __floordiv__(self, other):
        #Returns the quotient of a polynomial division
        return self.__truediv__(other)[0]


    def derivative(self):
        lst = self.get_coeff()[1:]
        for i in range(1,len(lst)+1):
            lst[i-1] = lst[i-1] * i
        return Polynomial(lst)

    def integral(self):
        lst = ["C"] + self.get_coeff()[:]
        for i in range(1, len(lst)):
            lst[i] = lst[i] * Fraction(1,i)
        return Polynomial(lst)

    #p(x)
    def evaluate(self, x):
        total = 0
        for i in range(self.len):
            total += x**i * self.get_coeff(i)
        return total

    def find_roots(self):
        #Using the Weierstrass method or Durand–Kerner method for simultaneous root finding
        #Multiply all the elements of a list together
        def mult_list_elements(lst):
            if len(lst) == 1:
                return lst[0]
            else:
                return lst[0] * mult_list_elements(lst[1:])
        #Divide polynomial by the coefficient of the highest power
        if self.get_coeff(-1) != 1:
            p = self * Fraction(1, self.get_coeff(-1))
        else:
            p = self
        #Setup: create p.deg roots
        roots = []
        for i in range(p.deg):
            #These roots are complex and chosen randomly - see Wikipedia page
            roots.append(complex(0.4,0.9)**i)
        #Root finding algorithm
        count = 0
        err1 = 1e-10
        while count <= 200:
            if count > 150:
                prev_roots = roots[:]
            for i in range(len(roots)):
                curr = roots[i]
                denom = [curr - x for x in roots if x != curr]
                denom_val = mult_list_elements(denom)
                roots[i] = curr - (p.evaluate(curr) / denom_val)
            count += 1
            if count > 151:
                #After 150 iterations, algorithm checks difference between prev and curr values of the roots. If they're small enough, exit.
                real_max = max([x.real-y.real for (x,y) in zip(roots, prev_roots)])
                imag_max = max([x.imag-y.imag for (x,y) in zip(roots, prev_roots)])
                if real_max < err1 and imag_max < err1:
                    count = 201
        #Simplify roots
        final_roots = []
        #Rounding and acceptable errors are based on the roots given by this algorithm to the Polynomial 1 - 3x + 3x^2 - x^3 = (1-x)^3
        #If errors are smaller, algorithm returns 1 - 2e-10j and 1.000000002 - 4e-10j as roots (which is wrong).
        round_val = 1e8
        err2 = 1e-7
        for root in roots:
            rounded_root = complex(round(root.real*round_val)/round_val, round(root.imag*round_val)/round_val)
            if abs(rounded_root.imag) <= err2:
                rounded_root = rounded_root.real
            if rounded_root not in final_roots:
                final_roots.append(rounded_root)
        self.roots = final_roots
        return final_roots

    #str and repr
    def __str__(self):
        poly = []
        poly.append(f"{self.get_coeff(0)}")
        for i in range(self.deg):
            next_num = self.get_coeff(i+1)
            if next_num < 0:
                next_x = f"- {abs(next_num)}x^{i+1}"
                poly.append(next_x)
            elif next_num > 0:
                next_x = f"+ {next_num}x^{i+1}"
                poly.append(next_x)
        return " ".join(poly)

    def __repr__(self):
        return f"Polynomial({str(self.get_coeff())})"

#Space
