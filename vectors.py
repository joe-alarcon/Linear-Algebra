from fractions import Fraction

fraction_limiting_denominator = 100000000

class Vector:
    zeros = False

    def __init__(self, identifier, how="build", v_lst=[]):
        self.name = identifier
        self.ent = []
        self.len = 0
        if how == "build":
            self.build_vector()
        else:
            self.create_vector(v_lst)
        self.dictionary_representations = {}

    def build_vector(self):
        #Build a vector from user input
        self.len = int(input("How many entries does your vector have? "))
        for i in range(self.len):
            new_entry_text = input(f"Enter the number for the {i+1}th entry: ")
            if "." in new_entry_text:
                new_entry = Fraction(float(new_entry_text))
            elif "/" in new_entry_text:
                new_entry = Fraction(new_entry_text)
            else:
                new_entry = int(new_entry_text)
            self.ent.append(new_entry)
        if all([x == 0 for x in self.ent]):
            self.zeros = True
        print(self.ent)

    def create_vector(self, v_lst):
        #Create a vector from a given list of real numbers
        self.len = len(v_lst)
        self.ent = v_lst
        if all([x == 0 for x in self.ent]):
            self.zeros = True

    def update(self, index, value):
        self.ent[index] = value
        if all([x == 0 for x in self.ent]):
            self.zeros = True

    def get(self, index=None):
        if type(index) is int:
            return self.ent[index]
        else:
            return self.ent

    #Vector Operations
    def __add__(self, other):
        assert self.len == other.len
        result_v = []
        for i in range(self.len):
            result_v.append(self.get(i)+other.get(i))
        return Vector(f"{self.name} + {other.name}", "create", result_v)

    def __sub__(self, other):
        assert self.len == other.len
        result_v = []
        for i in range(self.len):
            result_v.append(self.get(i)-other.get(i))
        return Vector(f"{self.name} + {other.name}", "create", result_v)

    def __eq__(self, other):
        return self.get() == other.get()

    def dot_product(self, other):
        assert self.len == other.len
        result = 0
        for i in range(self.len):
            mult = self.get(i)*other.get(i)
            if isinstance(mult, Fraction):
                mult.limit_denominator(fraction_limiting_denominator)
            result += mult
        return result

    def scale_vector(self, scale):
        result_v = []
        if type(scale) is not complex:
            frac_scale = Fraction(scale)
        else:
            frac_scale = scale
        for x in range(self.len):
            r = frac_scale*self.get(x)
            if type(r) is Fraction:
                k = r.limit_denominator(fraction_limiting_denominator)
                if k.denominator == 1:
                    k = k.numerator
            else:
                k = r
            if type(k) is not complex:
                rounded = round(k)
                if abs(rounded - k) < 1e-10:
                    k = rounded
                result_v.append(k)
            else:
                result_v.append(k)
        return Vector(f"{self.name} scaled by {frac_scale}", "create", result_v)

    def norm(self):
        result = 0
        for el in self.get():
            result += el**2
        sqrootnum = result ** 0.5
        if sqrootnum - int(sqrootnum) == 0:
            return int(sqrootnum)
        else:
            error = 1e9
            result = round(sqrootnum*error)/error
            return result

    def normalise(self):
        return self.scale_vector(1/self.norm())

    #Vector represented in a different basis
    def change_of_basis(self, basis):
        if f"{basis.name}" in self.dictionary_representations:
            print("This computation has already been done. Returning solution.")
            return self.dictionary_representations[f"{basis.name}"]
        mod_basis = basis[:]
        mod_basis.append(self)
        mod_basis_list = [vector.get() for vector in mod_basis]
        solved_matrix = Matrix(f"Matrix of {self.name}", "create", mod_basis_list, False)
        solved_matrix.row_reduce()
        self.dictionary_representations[f"{basis.name}"] = solved_matrix.reduced_columns[-1]
        return solved_matrix.reduced_columns[-1]

    #str and repr
    def __str__(self):
        grid_line = []
        for x in self.get():
            grid_line.append(str(x))
        return " ".join(grid_line)

    def __repr__(self):
        grid_line = []
        for x in self.get():
            grid_line.append(str(x))
        return " & ".join(grid_line)

    #copy
    def copy(self):
        copy_ent = self.get().copy()
        return Vector(f"Copy of {self.name}", "create", copy_ent)

class ZeroVector(Vector):
    zeros = True

    def __init__(self, length):
        self.len = length
        self.name = f"Zero vector {self.len}"
        self.ent = [0 for _ in range(self.len)]


#Space
