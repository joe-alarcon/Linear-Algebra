# Linear-Algebra
A program in Python for some of the most important aspects/computations of Linear Algebra using OOP.

The Matrix class in matrix.py. A matrix instance can be built using a 2D list of numbers or by user input. The user input can be any mathematical expression
that evaluates to a real or complex number. The program then takes each inner list and converts them to Vector instances and depending on what information
the inner lists hold, these vectors become part of the rows or columns. At the end, it does a transpose computation to store the columns or rows. This is
useful since row reduction is done by rows while other computations are done by columns. When the matrix is created, it determines if the matrix is square
and if it is symmetric. If the matrix is augmented, __init__ will also provide a copy of the matrix instance without the augmented column which proves to
be useful for many computations.

Once the instance is created, matrix_instance.run_all() will get most attributes and run most computations that can be done on a matrix. It row reduces
the matrix, which provides a solution if the matrix is augmented and can also perform a least squares solution if the matrix is not consistent, and also
provides the pivot positions which are used to determine linear independence and consistency. It also determines if the columns of the matrix are orthogonal
or orthonormal, which determines if the matrix is orthogonal. Determines if the matrix is invertible and if it is, it inverts the matrix and assigns this
matrix object to a parameter of the matrix instance, and it calculates the determinant. Then, it finds the eigenvalues and eigenvectors for square matrices,
and diagonalizes the matrix if the eigenvectors are a basis and stores the factorization as PDP^{-1}. Finally, it computes the singular value decomposition
of the matrix instance and stores this factorization as UZV^T.

To do all this, the vectors.py and polynomials.py files are imported. The Vector class allows to sum, subtract, find the standard dot product, find the
norm, normalise, and scale the vectors expressed as the rows and columns of a matrix. This is crucial for all the computations for the Matrix class. The
Polynomial class also has addition, subtraction, multiplication, and division, but its most important function is root finding when computing the
eigenvalues of a matrix. The algorithm this program uses is the Weierstrass method or Durand–Kerner method that uses complex initial values for simultaneous
root finding.

Finally, the VectorSet class in matrix.py is used to compute Gram Schmidt on a basis of vectors - especially useful for SVD. However, it can also be used
to find a change of basis for a given vector instance.

There is still much more to program such as orthogonal projections, standard matrices with respect to two different bases, change of bases matrices, and
possibly even more abstract computaions dealing with Linear Transformations.
