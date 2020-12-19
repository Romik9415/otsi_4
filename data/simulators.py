# import simpy
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

pv_specs = {'P_max': 310.0, 'V_mp': 32.6, 
			'I_mp': 9.51, 'U_oc': 40.7, 
			'I_sc': 9.8, 'U_wmax': 1500.0, 
			'I_maxfuse': 15.0, 'eta': 0.1905, 
			'cells_in_array': 60, 
			'cell_size': (156, 156)}

acc_specs = {'capacity': 13.5*1e3}

max_p = 975 #0.975
mean_p = 400 #0.4 # Wh
std_p = 120 #0.12
min_p = 160 #0.16

class Grid:

	def __init__(self, power,
		num_consumers, generators=None):

		self.power = power
		if generators is None:
			self.generators = [model_pv for _ in range(300)]
		else:
			self.generators = generators
		self.num_consumers = num_consumers

		self.populate_consumers()

	#def populate_generators(self):

	def populate_consumers(self):

		self.population = []
		for consumer in range(self.num_consumers):
			self.population.append(Consumer())

	def simulate(self, start_time=datetime.now(), num_hours=8760):
		result = pd.DataFrame()
		loads = np.empty(shape=num_hours)
		generation = np.empty(shape=num_hours)
		for i, h in enumerate(tqdm(range(num_hours))):
			total_load = 0
			total_gen = 0
			for c in self.population:
				total_load += c.load_grid(h)
			for g in self.generators:
				specs = pv_specs
				total_gen += g(specs, rad_coefs[h%24])
			loads[i] = total_load
			generation[i] = total_gen
			result = result.append(
				{'time': datetime.strftime(
					start_time, format='%d-%b %H')},
				ignore_index=True)
			start_time += timedelta(hours=1)
		
		result['load'] = loads
		result['gen'] = generation
		result = result.set_index('time')
		#print(result.sum()/1e6)
		return result

class PV_module:
	def __init__(self):
		pass

class Battery:
	def __init__(self, capacity=13.8*1e3, min_cap=13.8*1e3*0.3):
		self.capacity = capacity
		self.min_cap = min_cap
		self.power = self.__init_power()

	def __init_power(self):
		return self.min_voltage

	def charge_discharge(self, energy):
		if self.power+energy > self.capacity:
			self.power = capacity
		elif self.power-energy < self.min_cap:
			self.power = self.min_cap

class Consumer:

	def __init__(self, min_load=min_p, 
		max_load=max_p, avg_load=mean_p, std=std_p):

		self.min_load = min_load
		self.max_load = max_load
		self.avg_load = avg_load
		self.std = std

		self.priors = self._init_priors()

	def _init_priors(self):

		priors_scaled = {0: 0.46, 1: 0.479, 2: 0.467, 
						3: 0.438, 4: 0.434, 5: 0.452, 
						6: 0.572, 7: 0.914, 8: 1.017, 
						9: 1.377, 10: 1.279, 11: 0.904, 
						12: 1.115, 13: 0.861, 14: 1.0743, 
						15: 1.065, 16: 1.443, 17: 1.583, 
						18: 1.703, 19: 1.48, 20: 1.33, 
						21: 1.512, 22: 1.302, 23: 0.749}

		return {k: v for k, v in priors_scaled.items()}

	def _add_small_noise(self, value, std_constrain, constant=0.09):
		more_or_less = np.random.choice([0, 1], size=1, p=[0.5, 0.5])
		if more_or_less == 0:
			return value+np.random.normal(value, self.std, 1)[0]*constant
		return value-np.random.normal(value, self.std, 1)[0]*constant

	def load_grid(self, hour):

		if hour >= 24:
			hour = hour % 24
		#print(self.max_load)

		max_load_prob = self.priors[hour]*0.1
		p_maxload = np.random.choice(
			[0, 1], size=1, p=[1-max_load_prob, max_load_prob])[0]
		if p_maxload==1:
			return self.max_load #*self.priors[hour]
		else:
			min_load_prob = self.priors[hour]*0.1
			p_minload = np.random.choice(
				[0, 1], size=1, p=[1-min_load_prob, min_load_prob])[0]
			if p_minload==1:
				return self.min_load #*self.priors[hour]
			p_normal = np.random.normal(self.avg_load, self.std, 1)[0]
			return self._add_small_noise(p_normal, self.std) #*self.priors[hour]


class NoiseSimulator:

	def __init__(self, mu=35, sigma=7, 
		min_=10, max_=80, 
		day_range='afternoon',
		coords=(0.0, 0.0),
		points=100):

		self.mu = mu
		self.sigma = sigma
		self.min_ = min_
		self.max_ = max_
		self.day_range = day_range
		self.coords = coords
		self.points = points

	def __len__(self):
		return self.points

	def __next__(self):

		for i in range(self.points):
			point = np.random.normal(self.mu, self.sigma, 1)[0]
			yield point
		

def model_pv(specs=pv_specs, rad_coef=0.6):
	# pv_area = specs['cell_size'][0]*specs['cell_size'][1]*specs['cells_in_array']
	return rad_coef*specs['P_max']

def model_wt():
	pass

def model_bs():
	pass
