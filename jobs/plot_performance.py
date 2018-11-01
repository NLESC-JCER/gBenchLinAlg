import os
import json
from collections import namedtuple
import numpy as np
import  matplotlib.pyplot as plt

BenchData = namedtuple('BenchData',['size','time','std'])

def get_num_procs(fname):

	if '_np' not in fname:
		np = 1
	else:
		np = int(fname.split('np')[1].split('_')[0])

	if '_omp' not in fname:
		omp = int(fname.split('_')[1].split('.')[0])
	else:
		omp = int(fname.split('_omp')[1].split('.')[0])

	return np, omp

def extract_data(fname):

	with open(fname) as f:
		data = json.load(f)
	bench = data['benchmarks']

	size = []
	time = []
	std = []
	for b in bench:

		name = b['name']
		s = int(name.split('/')[1])

		if s not in size:
			size.append(s) 

		if name.endswith('real_time_mean'):
			time.append(float(b['real_time']))

		if name.endswith('real_time_stddev'):
			std.append(float(b['real_time']))
		
	return BenchData(size,time,std)

def get_data_folder(folder_name):

	data = dict()
	for f in os.listdir(folder_name):
		if f.endswith('.out'):
			np,omp = get_num_procs(f)
			fname = folder_name +  '/' + f

			if np not in data:
				data[np] = dict()

			data[np][omp] = extract_data(fname)
	return data

def plot_timeVSnumThread(name,nump=1,size=-1,legend=''):

	time = []
	nomp = list(data[name][nump].keys())

	for k in nomp:
		time.append(data[name][nump][k].time[size])
	mat_size = data[name][nump][k].size[size]

	indx = np.argsort(nomp)
	nomp = np.log10(np.array(nomp)[indx])
	time = np.log10(np.array(time)[indx])

	plt.plot(nomp,time,'o-')

	if legend == '':
		legend = name.upper() + ' ' +  str(nump) + ' procs (%d)' %mat_size

	return legend

def plot_timeVSnumProc(name,numthread=1,size=-1,legend=''):

	time = []
	possible_numproc = list(data[name].keys())
	numproc = []

	for k in possible_numproc:
		if numthread in data[name][k].keys():
			mat_size = data[name][k][numthread].size[size]
			time.append(data[name][k][numthread].time[size])
			numproc.append(k)

	indx = np.argsort(numproc)
	numproc = np.log10(np.array(numproc)[indx])
	time = np.log10(np.array(time)[indx])

	plt.plot(numproc,time,'o-')

	if legend == '':
		legend = name.upper() + ' ' + str(numthread) + ' threads (%d)' %mat_size

	return legend

if __name__ == "__main__":

	folders = ['arma','eigen','cblas','elmpi']
	data = dict()
	for f in folders:
		data[f] = get_data_folder(f)


	legends = []
	fig = plt.figure()
	ax = fig.add_subplot(111)

	legends.append(plot_timeVSnumThread('arma',size=-2))
	legends.append(plot_timeVSnumThread('eigen',size=-2))
	legends.append(plot_timeVSnumProc('elmpi',size=-2))
	legends.append(plot_timeVSnumProc('elmpi',numthread=2,size=-2))
	legends.append(plot_timeVSnumProc('elmpi',numthread=4,size=-2))

	plt.xlabel('Number Procs/Threads')
	plt.ylabel('Time (s)')
	plt.xticks(np.log10(np.array([1,2,4,8,16])),['1','2','4','8','16'])
	#plt.yticks(np.log10(np.array([5*10**3,10**4,5*10**4])),['5','10','50'])
	plt.yticks(np.log10(np.array([10**3,5*10**3,10**4])),['1','5','10'])
	plt.legend(legends)
	plt.title('GEMM 4096 x 4096')
	plt.show()

