from math import log
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.stats import zipfian

# indices = [Na, E, Ha, J, F, E_heip, E_1mD, E_1dD, E_lnD, G_21, O, E_x, E_mcl, 
# 		   E_prime, E_var]

b = 2

def normalize(p):
	if np.isclose(sum(p), 1):
		return p
	return np.array(p) / sum(p)

def Na(a, p):
	p = normalize(p)
	if a == 1:
		return 2 ** (- sum(x * log(x, b) if x != 0 else 0 for x in p))
	try :
		return sum(x ** a for x in p) ** (1 / (1 - a))
	except ZeroDivisionError:
		return 0

def E(n, m, p):
	p = normalize(p)
	try :
		return Na(n, p) / Na(m, p)
	except ZeroDivisionError:
		return 0

def Ha(a, p):
	p = normalize(p)
	if a == 1:
		return H(p)
	return (1 / 1 - a) * log(sum(x ** a for x in p), b)

def H(p):
	p = normalize(p)
	return - sum(x * log(x, b) for x in p)

def J(p):
	p = normalize(p)
	return H(p) / log(Na(0, p), b)

def F(n, m, p):
	p = normalize(p)
	try :
		return (Na(n, p) - 1) / (Na(m, p) - 1)
	except ZeroDivisionError:
		return 0
	
def E_heip(p):
	p = normalize(p)
	return F(1, 0, p)

def E_1mD(p):
	p = normalize(p)
	try :
		return (1 - (Na(2, p) ** -1)) / (1 - (Na(0, p) ** -1))
	except ZeroDivisionError:
		return 0
	
def E_1dD(p):
	p = normalize(p)
	return E(2, 0, p)

def E_lnD(p):
	p = normalize(p)
	return (- (log(Na(2, p) ** -1, b))) / log(Na(0, p), b)

def G_21(p):
	p = normalize(p)
	f = F(2, 1, p)
	if f <= (0.5 ** 0.5):
		return f ** 3
	else:
		return 0.636611 * f * np.arcsin(f)
	
def O(p):
	p = normalize(p)
	return sum(min(x, Na(0, p) ** -1) for x in p)

def E_x(p):
	p = normalize(p)
	return (O(p) - (Na(0, p) ** -1)) / (1 - (Na(0, p) ** -1))

def E_mcl(f):
	N = sum(f)
	return (N - ((sum(x ** 2 for x in f)) ** 0.5)) / (N - (N / (Na(0, f) ** 0.5)))

def E_prime(p):
	p = normalize(p)
	return 1 - ((sum(sum(np.abs(s1 - s2) for s2 in p[x+1:]) for x, s1 in enumerate(p[:-1]))) / (Na(0, p)))

def E_var(f):
	return 1 - (2 / np.pi) * np.arctan((sum((log(s) - sum(log(t) for t in f) / Na(0, f)) ** 2 for s in f)) / (Na(0, f)))

def zipf_s(p):
	return curve_fit(
		lambda x, s: zipfian.pmf(x, s, len(p)),
		list(range(1, len(p)+1)),
		p.sort_values(ascending=False)
	)[0][0]

def zipf_s_n(p):
	counts = np.sort(np.array(p))[::-1]
	y_data = counts / np.sum(counts)
	# print(counts, y_data)

	def objective(params):
		s, n = params
		n = int(round(n)) 
		padded_x_data = np.arange(1, n+1)
		padded_y_data = np.concatenate([
			y_data,
			np.zeros(n- len(y_data))
		])
		pmf_vals = zipfian.pmf(padded_x_data, s, n)
		if np.any(np.isnan(pmf_vals)) or np.any(np.isinf(pmf_vals)):
			return np.inf
		return np.sum((pmf_vals - padded_y_data)**2)

	result = minimize(
		objective,
		x0=[1.0, len(p)],  # initial guess
		method='Nelder-Mead',
		bounds=[(0,10), (len(p), 1e6)],
		options={'maxiter': 50}
	)


	s, n = result.x
	p_value = zipf_p_value(counts, s, n)

	return s, n, p_value

def zipf_p_value(p, s, n):
	counts = np.sort(np.array(p))[::-1]
	# y_data = counts / np.sum(counts)
	ranks = np.arange(1, len(counts) + 1)
	samples = np.repeat(ranks, counts) 


	n = int(round(n))
	null_dist = zipfian.pmf(ranks, s, n)
	ll_obs = np.sum(counts * np.log(null_dist))
	iterations = 10000
	sim_counts = np.random.multinomial(len(samples), null_dist, size=iterations)
	ll_sim = sim_counts @ np.log(null_dist)

	# print('LL OBS:', ll_obs)
	# print('LL SIM:', np.mean(ll_sim))
	# print('R²:', np.sum((null_dist - (counts / np.sum(counts))) ** 2))

	p_value = np.mean(ll_sim <= ll_obs)
	return p_value


def true_pred_to_dist(true_pred) -> tuple[np.ndarray, int]:
	tp_grpby = true_pred.assign(n=1).groupby(0).count()['n']
	n = sum(tp_grpby)
	return tp_grpby / n, n

def diversity_eval(true_pred):
	tp_grpby, n = true_pred_to_dist(true_pred)
	
	est_s = zipf_s(tp_grpby)

	return {
		'richness': Na(0, tp_grpby),
		'N1': Na(1, tp_grpby),
		'normalize_r': Na(0, tp_grpby)/n,
		'H': Ha(1, tp_grpby),
		'J': J(tp_grpby),
		'e10': E(1, 0, tp_grpby),
		'e21': E(2, 1, tp_grpby),
		'e20': E(2, 0, tp_grpby),
		'F10': F(1, 0, tp_grpby),
		'F21': F(2, 1, tp_grpby),
		'F20': F(2, 0, tp_grpby),
		'E_heip': E_heip(tp_grpby),
		'E_1mD' : E_1mD(tp_grpby), 
		'E_1dD' : E_1dD(tp_grpby), 
		'E_lnD' :E_lnD(tp_grpby), 
		'G_21' : G_21(tp_grpby), 
		'O': O(tp_grpby), 
		'E_x': E_x(tp_grpby), 
		'E_mcl': E_mcl(tp_grpby), 
		'E_prime' : E_prime(tp_grpby), 
		'E_var' : E_var(tp_grpby),
		'1/S': 1/est_s,
		'exp_minus_S': np.exp(-est_s),
		's_p_val': zipf_p_value((tp_grpby * n).astype(int), est_s, len(tp_grpby)),
		'zipf_s_n': zipf_s_n((tp_grpby * n).astype(int))
	}