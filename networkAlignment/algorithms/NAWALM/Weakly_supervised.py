import numpy as np

W_m = None
W_d = None
T_tgt = None
T_src = None
def P_e(gamma, x_src, x_tgt, Z):
	"""
	P_e(e_tgt| e_src): The probability that source entity e_src is aligned to target entity e_tgt
	equation (1)
	"""
	return np.exp(-gamma * np.sum((np.matmul(x_src, W_m) - x_tgt)**2)) / Z


"""
Discriminator D(x): R_1xd -> R: The probability that x is a embedding of target
"""

def A(t):
	"""
	Return Aligned vector 
	"""
	return np.matmul(t, W_m)



def Dt(t):
	"""
	probability that t is target entity
	"""
	return np.matmul(t, W_d)


def O_Dt(W_d):
	E1 = np.mean(np.array([np.log(Dt(t_tgt)) for t_tgt in T_tgt]))
	E2 = np.mean(np.array([np.log(1 - Dt(A(t_src))) for t_src in T_src]))

if __name__ == "__main__":
	d = 4
	W = np.random.rand(d, d)
	Z = 1
	gamma = 1
	x_src = np.random.rand(1, d)
	x_tgt = np.random.rand(1, d)

	print(P_e(gamma, W, x_src, x_tgt, Z))