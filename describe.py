#!/usr/bin/env python3
import sys
import math
import pandas as pd
import numpy as np
from functools import reduce
from libs import snackbar

def describe(arr):
	def percentile(array, percent):
		k = (len(array) - 1) * percent
		f = math.floor(k)
		c = math.ceil(k)
		if f == c:
			return array[int(k)]
		d0 = array[int(f)] * (c - k)
		d1 = array[int(c)] * (k - f)
		return d0 + d1

	count = 0
	mini = float('inf')
	maxi = -float('inf')
	mean = 0
	values = []
	for x in arr:
		if(not np.isnan(x)):
			count += 1
			mean += x
			values.append(x)
		if(maxi < x):
			maxi = x
		if(mini > x):
			mini = x
	if(count == 0):
		return [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
	mean /= count

	std = 0
	for x in values:
		std += (x - mean) ** 2
	std = (std / (count - 1)) ** 0.5
	values = sorted(values)
	per_25 = percentile(values, 0.25)
	per_50 = percentile(values, 0.5)
	per_75 = percentile(values, 0.75)
	return [count, mean, std, mini, per_25, per_50, per_75, maxi]


def read():
	df = pd.DataFrame()
	if len(sys.argv) ==1 :
		snackbar("Set file name", 'error')
	try:
		name_file = sys.argv[1]
		df = pd.read_csv(name_file)
	except OSError:
		snackbar("Can\'t open file ({name})".format(name=name_file), 'error')
	except (ValueError, IndexError):
		snackbar("Not valid file ({name})".format(name=name_file), 'error')
	except Exception:
		snackbar("Unknown error", 'info')
	df_types = df.select_dtypes(include=['float64', 'int64'])
	names = df_types.keys()
	statistics = list(map(lambda name: describe(df_types[name]), names))
	name_stats = ['count', 'means', 'std', 'min', '25%', '50%', '75%', 'max']
	result = pd.DataFrame(statistics, columns=[name_stats], index=names).T
	return result


def main():
    print(read())


if __name__ == '__main__':
    main()
