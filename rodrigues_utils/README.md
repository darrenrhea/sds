# rodigues_utils

The Rodrigues angle-axis vector is important to us
because it parametrizes the group of rotation matrices SO_3,
(Special Orthogonal 3x3 Matrices, i.e. 3x3 matrices M s.t. det(M) = 1 and M^T M = I)
with only 3 parameters.
The axis of the rotation is the Rodrigues vector normalized.
The norm of the Rodrigues vector is the amount of radians to rotate
about that axis in right-hand-rule sense.
This is to say that all SO_3 matrices M actually rotate about some axis,
namely the eigenvector of M of eigenvalue 1.

