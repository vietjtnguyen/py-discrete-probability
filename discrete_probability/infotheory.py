from math import log

from discrete_probability import Assignment

def entropy(distr, variables):
	ent = 0.0
	assignments = Assignment.generate(variables)
	for assignment in assignments:
		proba = distr(*assignment)
		if proba > 0.0:
			ent -= proba * log(proba, 2)
	return ent

def conditional_entropy(distr, X, Y):
	ent = 0.0
	for x in X.assignments:
		for y in Y.assignments:
			proba_xy = distr(x, y)
			proba_x_y = distr(x | y)
			if proba_xy > 0.0 and proba_x_y > 0.0:
				ent -= proba_xy * log(proba_x_y, 2)
	return ent

def mutual_information(distr, X, Y):
	mi = 0.0
	for x in X.assignments:
		for y in Y.assignments:
			proba_x = distr(x)
			proba_y = distr(y)
			proba_xy = distr(x, y)
			if proba_x > 0.0 and proba_y > 0.0 and proba_xy > 0.0:
				mi += proba_xy * log(proba_xy / (proba_x * proba_y), 2)
	return mi

def kl_divergence(distr_l, distr_r, variables):
	kl = 0.0
	assignments = Assignment.generate(variables)
	for assignment in assignments:
		proba_l = distr_l(*assignment)
		proba_r = distr_r(*assignment)
		if proba_l > 0.0 and proba_r > 0.0:
			kl += proba_l * log(proba_l / proba_r, 2)
	return kl

