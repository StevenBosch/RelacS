def getDefaults(tau, **kwargs):
	return {
		'FBR_scope': kwargs.get('fbr_scope', [-5, 5]), # relation to size of fluctuations of noise
		'FBR_range': kwargs.get('fbr_range', [-20, 20]), # relation to adjusting tau when looking at distance between fore- and background in dB
		'delta_time': kwargs.get('delta_time', .005), #seconds
		'tau': tau, #seconds
		'noiseSTD': kwargs.get('noise_std', 3.),
		'step': kwargs.get('step', .3),
		'subtract': kwargs.get('subtract', None),
        'mask': kwargs.get('mask', None)
	}

defaults = getDefaults(0.0)

def tau(t, end):
	while t < end:
		t *= 2
		yield t

def defmodels(t,end):
	defaultmodels = [getDefaults(t) for t in tau(t, end)]
	return defaultmodels
