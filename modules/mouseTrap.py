#!/usr/bin/env python3

import numpy as np
from numpy import pi, sin, cos, arctan
from obspy import Stream, Trace, UTCDateTime
import matplotlib.pyplot as plt
from math import sin, cos, radians

def next_power_of_2(n):
	"""
	Return next power of 2 greater than or equal to n
	"""
	return 2**(n-1).bit_length()


class mouse:
	def __init__(self, min_step = 2., min_step2 = 0.1, fit_time_before = 115, fit_time_after = 60):
		self.min_step        = min_step
		self.min_step2       = min_step2
		self.fit_time_before = fit_time_before
		self.fit_time_after  = fit_time_after

	def create(self, paz, length, dt, onset, typ=1, demean=True, integrate=True):
		self.nmax = nmax = next_power_of_2(length)
		self.df = df = 1. / (nmax*dt)
		self.dt = dt
		self.onset = onset
		self.typ = typ
		mys = self.mouse = np.zeros(nmax, dtype=complex)
		mys[0] = 0+0j
		for i in range(1, nmax//2 + 1): # negative frequencies nmax/2+1 .. nmax-1 are complex conjugate of positive frequencies
			f = i*df
			if (typ == 2):
				mys[i] = (1./dt) / (2*np.pi*1j*f) # input velocity= step (i.e. input accel   = delta); mouse II
			elif (typ == 3):
				mys[i] = 1. / dt # mouse type III
			else:
				mys[i] = (1./dt) / (2.*np.pi*1j*f)**2. # input velocity= linear  (i.e. input accel = step); mouse I
				# INPUT ACCEL STEP = 1 m/s^2;
			mys[i] = mys[i] * np.exp(-2*np.pi*1j*f*onset) # time shift
		# Apply effect of the instrument:
		for i in range(nmax):
			fc = i * df * 2 * np.pi * 1j
			H = 1.+0j
			for zero in paz.zeros:
				H *= (fc - zero)
			for pole in paz.poles:
				H /= (fc - pole)
			H *= paz.gain * paz.sensitivity
			mys[i] *= H
		for i in range(1, nmax//2):
			mys[nmax-i] = np.conj(mys[i]) # DFT of the real signal has complex conjugate spectrum
		self.mouse = np.fft.ifft(mys)
		if demean:
			self.demean()
		if integrate:
			self.integrate()


	def create_from_template(self, template, t_template, onset_time=0.0,
							 demean=False, integrate=False):
		"""
		Use an externally generated mouse/glitch template instead of building
		one from a single poles-and-zeros response.

		This is useful when the template has already been generated from the
		complete instrument response (for example from StationXML/evalresp).

		Parameters
		----------
		template : array_like
			One-dimensional template waveform. It must be in the same data domain
			and units as the Stream/Trace supplied later to ``fit_3D``/``fit_mouse``.
		t_template : array_like
			Time axis corresponding to ``template`` in seconds. The axis must be
			uniformly sampled and increasing.
		onset_time : float, optional
			Time value in ``t_template`` corresponding to the acceleration-step
			onset. For a template whose time axis is relative to the step, this is
			normally 0.0 s.
		demean : bool, optional
			Apply the original MouseTrap pre-onset demeaning to the template.
			Default False because externally generated templates are usually
			already baseline corrected.
		integrate : bool, optional
			Integrate the template once after loading. Default False. Leave this
			False when fitting the raw-count template to raw-count data.
		"""
		template = np.asarray(template).squeeze()
		t_template = np.asarray(t_template, dtype=float).squeeze()

		if template.ndim != 1 or t_template.ndim != 1:
			raise ValueError('template and t_template must both be one-dimensional.')
		if len(template) != len(t_template):
			raise ValueError(
				f'template and t_template must have the same length; got '
				f'{len(template)} and {len(t_template)}.'
			)
		if len(template) < 2:
			raise ValueError('template must contain at least two samples.')

		dt_all = np.diff(t_template)
		if np.any(dt_all <= 0):
			raise ValueError('t_template must be strictly increasing.')

		dt = float(np.median(dt_all))
		if not np.allclose(dt_all, dt, rtol=1e-5, atol=max(1e-12, abs(dt)*1e-8)):
			raise ValueError('t_template must be uniformly sampled.')

		onset_index = int(np.argmin(np.abs(t_template - onset_time)))
		onset_from_start = float(t_template[onset_index] - t_template[0])

		self.nmax = len(template)
		self.df = 1.0 / (self.nmax * dt)
		self.dt = dt
		self.onset = onset_from_start
		self.typ = 'external_template'
		self.mouse = template.astype(np.complex128, copy=True)

		# The fitting routines use a window around self.onset. Fail early with
		# a clear message if the supplied template cannot cover that window.
		template_duration = (len(template) - 1) * dt
		if self.onset < self.fit_time_before:
			raise ValueError(
				f'Template has only {self.onset:.3f} s before the onset, but '
				f'fit_time_before={self.fit_time_before:.3f} s. Either provide a '
				f'longer template or construct mouse(..., fit_time_before=...).'
			)
		if self.onset + self.fit_time_after > template_duration:
			after_available = template_duration - self.onset
			raise ValueError(
				f'Template has only {after_available:.3f} s after the onset, but '
				f'fit_time_after={self.fit_time_after:.3f} s. Either provide a '
				f'longer template or construct mouse(..., fit_time_after=...).'
			)

		if demean:
			self.demean()
		if integrate:
			self.integrate()

	def demean(self):
		last = int(self.onset / self.dt - 10) # take 10 samples less to avoid possible noc-causal blend of the mouse onset
		sum = 0
		for i in range(last):
			sum += self.mouse[i]
		mean = sum/last
		self.mouse -= mean

	def integrate(self):
		"""
		Integrate the synthetic mouse using cumulative sum (:func:`~numpy.cumsum`).
		"""
		self.mouse = self.mouse.cumsum()
		self.mouse *= self.dt


	def fit_3D(self, st, t_min=0, t_max=0):
		self.fit_mouse(st, t_min, t_max)

	def fit_mouse(self, st, t_min=0, t_max=0):
		"""
		Fit the mouse over the requested data interval.

		The original MouseTrap search allowed candidate onsets right at the
		start/end of a record, even though the fit itself uses data before and
		after the candidate onset.  That can index one sample beyond the data
		array when fitting a complete record.  Here the search is restricted to
		candidate onsets for which the full [fit_time_before, fit_time_after]
		window exists.  No data are otherwise discarded.
		"""
		dt = self.dt

		# ------------------------------------------------------------
		# Input checks and component extraction
		# ------------------------------------------------------------
		if isinstance(st, Stream):
			if len(st) != 3:
				raise ValueError('3-D MouseTrap fitting requires exactly three traces.')

			lengths = [len(tr) for tr in st]
			if len(set(lengths)) != 1:
				raise ValueError(f'All three traces must have the same length; got {lengths}.')

			for tr in st:
				if not np.isclose(tr.stats.delta, dt, rtol=1e-7, atol=1e-12):
					raise ValueError(
						f'Template dt={dt:g} s does not match {tr.id} dt={tr.stats.delta:g} s.'
					)

			self.reclen = length = lengths[0]
			N = E = Z = None
			for tr in st:
				comp = tr.stats.channel[-1]
				if comp == 'N':
					N = tr.data
				elif comp == 'E':
					E = tr.data
				elif comp == 'Z':
					Z = tr.data
				else:
					raise ValueError(f'Unknown component {comp}; channels must end in N, E, Z.')
			if N is None or E is None or Z is None:
				raise ValueError('Stream must contain one N, one E, and one Z component.')

		elif isinstance(st, Trace):
			if not np.isclose(st.stats.delta, dt, rtol=1e-7, atol=1e-12):
				raise ValueError(
					f'Template dt={dt:g} s does not match {st.id} dt={st.stats.delta:g} s.'
				)
			self.reclen = length = len(st)
			D = st.data
		else:
			raise TypeError('Parameter st must be an ObsPy Stream or Trace.')

		if length < 2:
			raise ValueError('Input record is too short.')

		# Coarse and fine search spacing, retaining the original MouseTrap
		# interpretation of min_step and min_step2.
		step = max(1, int(self.min_step / dt)) if dt < self.min_step else 1
		step2 = max(1, int(self.min_step2 / dt)) if dt < self.min_step2 else 1

		# ------------------------------------------------------------
		# Safe onset interval
		# ------------------------------------------------------------
		record_end = (length - 1) * dt
		requested_min = 0.0 if t_min is None or t_min < 0 else float(t_min)
		requested_max = record_end if t_max is None or t_max <= 0 else float(t_max)

		requested_min = max(0.0, requested_min)
		requested_max = min(record_end, requested_max)

		# A candidate can only be tested when its complete fitting window is
		# inside the data record.
		safe_min = max(requested_min, float(self.fit_time_before))
		safe_max = min(requested_max, record_end - float(self.fit_time_after))

		if safe_max < safe_min:
			raise ValueError(
				'No valid MouseTrap onset remains after applying the fitting window: '
				f'requested [{requested_min:.3f}, {requested_max:.3f}] s, '
				f'fit_time_before={self.fit_time_before:.3f} s, '
				f'fit_time_after={self.fit_time_after:.3f} s, '
				f'record_end={record_end:.3f} s.'
			)

		onset_min = int(np.ceil(safe_min / dt - 1e-12))
		onset_max = int(np.floor(safe_max / dt + 1e-12))

		self.a = np.zeros(length)
		self.fit = -np.inf
		self.shift = None
		if isinstance(st, Stream):
			self.phi = np.zeros(length)
			self.theta = np.zeros(length)

		def _evaluate_onset(onset_index):
			# Preserve the original internal shift convention:
			# onset_found = (length - shift) * dt.
			shift = length - int(onset_index)
			if isinstance(st, Stream):
				self.invert(N, E, Z, shift)
			else:
				self.invert_1D(D, shift)

		# ------------------------------------------------------------
		# Coarse grid over the FULL requested stretch
		# ------------------------------------------------------------
		coarse_indices = list(range(onset_min, onset_max + 1, step))
		if coarse_indices[-1] != onset_max:
			coarse_indices.append(onset_max)

		for onset_index in coarse_indices:
			_evaluate_onset(onset_index)

		if self.shift is None:
			raise RuntimeError('MouseTrap could not evaluate any valid candidate onset.')

		# ------------------------------------------------------------
		# Fine grid around the best coarse solution
		# ------------------------------------------------------------
		if step2 < step:
			best_onset = length - self.shift
			fine_min = max(onset_min, best_onset - step + step2)
			fine_max = min(onset_max, best_onset + step - step2)
			for onset_index in range(fine_min, fine_max + 1, step2):
				_evaluate_onset(onset_index)

		self.onset_found = (length - self.shift) * dt
		self.amplitude = self.a[self.shift]
		if isinstance(st, Stream):
			self.phi = self.phi[self.shift]
			self.theta = self.theta[self.shift]

	def invert(self, N, E, Z, shift):
		"""Evaluate one candidate onset for a three-component fit."""
		dt = self.dt
		mys = self.mouse
		t_mouse = (self.reclen - shift) * dt

		fit_t0 = int(round((t_mouse - self.fit_time_before) / dt))
		fit_t1 = int(round((t_mouse + self.fit_time_after) / dt))
		on = self.reclen - int(round(self.onset / dt))

		m0 = fit_t0 + shift - on
		m1 = fit_t1 + shift - on

		# A valid candidate must have the full fitting window in both the
		# data and the external/synthetic mouse template.
		if fit_t0 < 0 or fit_t1 > self.reclen or m0 < 0 or m1 > len(mys):
			return

		g = np.asarray(mys[m0:m1].real, dtype=float)
		n = np.asarray(N[fit_t0:fit_t1], dtype=float)
		e = np.asarray(E[fit_t0:fit_t1], dtype=float)
		z = np.asarray(Z[fit_t0:fit_t1], dtype=float)

		if len(g) == 0 or not (len(g) == len(n) == len(e) == len(z)):
			return

		# determination of phi: preserve the algebra used by the original code
		ms_N = np.dot(g, n)
		ms_E = np.dot(g, e)
		if ms_N == 0 and ms_E == 0:
			return
		if ms_E < ms_N:
			ph = arctan(ms_E / ms_N) if ms_N != 0 else -pi/2
		else:
			ph = pi/2 - arctan(ms_N / ms_E) if ms_E != 0 else 0.0

		# determination of theta
		h = n*cos(ph) + e*sin(ph)
		ms_Z = np.dot(g, z)
		ms_NE = np.dot(g, h)
		if ms_Z == 0 and ms_NE == 0:
			return
		if ms_Z < ms_NE:
			th = arctan(ms_Z / ms_NE) if ms_NE != 0 else -pi/2
		else:
			th = pi/2 - arctan(ms_NE / ms_Z) if ms_Z != 0 else 0.0
			if th > pi/2:
				th -= pi

		# amplitude of acceleration step
		proj = n*cos(ph)*cos(th) + e*sin(ph)*cos(th) + z*sin(th)
		mm = np.dot(g, g)
		if not np.isfinite(mm) or mm <= 0:
			return
		a = np.dot(g, proj) / mm

		# variance-reduction-like fit used by the original MouseTrap code
		model_N = g*a*cos(ph)*cos(th)
		model_E = g*a*sin(ph)*cos(th)
		model_Z = g*a*sin(th)
		fit_nom = (
			np.dot(n-model_N, n-model_N)
			+ np.dot(e-model_E, e-model_E)
			+ np.dot(z-model_Z, z-model_Z)
		)
		fit_denom = np.dot(n, n) + np.dot(e, e) + np.dot(z, z)
		if not np.isfinite(fit_denom) or fit_denom <= 0:
			return
		fit = 1.0 - fit_nom / fit_denom

		if a < 0:
			a = -a
			th = -th
			ph += np.pi

		self.a[shift] = a
		self.phi[shift] = ph % (2*np.pi)
		self.theta[shift] = th

		if np.isfinite(fit) and fit > self.fit:
			self.fit = fit
			self.shift = shift

	def invert_1D(self, D, shift):
		"""Evaluate one candidate onset for a one-component fit."""
		dt = self.dt
		mys = self.mouse
		t_mouse = (self.reclen - shift) * dt
		fit_t0 = int(round((t_mouse - self.fit_time_before) / dt))
		fit_t1 = int(round((t_mouse + self.fit_time_after) / dt))
		on = self.reclen - int(round(self.onset / dt))
		m0 = fit_t0 + shift - on
		m1 = fit_t1 + shift - on

		if fit_t0 < 0 or fit_t1 > self.reclen or m0 < 0 or m1 > len(mys):
			return

		g = np.asarray(mys[m0:m1].real, dtype=float)
		d = np.asarray(D[fit_t0:fit_t1], dtype=float)
		if len(g) == 0 or len(g) != len(d):
			return

		mm = np.dot(g, g)
		if not np.isfinite(mm) or mm <= 0:
			return
		a = np.dot(g, d) / mm
		model = g*a
		fit_denom = np.dot(d, d)
		if not np.isfinite(fit_denom) or fit_denom <= 0:
			return
		fit = 1.0 - np.dot(d-model, d-model) / fit_denom

		self.a[shift] = a
		if np.isfinite(fit) and fit > self.fit:
			self.fit = fit
			self.shift = shift

	def exist(self, Ts=120, likehood=False):
		likehood = self.fit - abs(self.onset_found - Ts) / 50
		if likehood > 0.7:
			return 2
		elif likehood > 0.2:
			return 1
		else:
			return 0
	
	def params(self, degrees=False):
		"""
		:type degrees: bool, optional
		:param degrees: return angles in degrees instead of radians
		
		Returns tuple of mouse onset (second), amplitude (m/s/s for mouse type 1), phi (radians, optionaly degrees), theta (radians, optionaly degrees), and fit (percent).

		If angles have not been determined, corresponding output values are ``None``.
		"""
		if degrees: c=180./np.pi
		else: c=1
		try:
			return (self.onset_found, self.amplitude, self.phi*c, self.theta*c, self.fit)
		except AttributeError:
			return (self.onset_found, self.amplitude, None, None, self.fit)
	
	def plot(self, st, outfile='', distance=5e3, ylabel='', xmin=100, xmax=300, legend=False, yaxes=True):
		npts = st[0].stats.npts
		samprate = st[0].stats.sampling_rate
		t = np.arange(0, npts / samprate, 1 / samprate)
		fit_t0 = self.onset_found - self.fit_time_before
		fit_t1 = self.onset_found + self.fit_time_after
		samp_0 = int((self.onset - self.fit_time_before) / self.dt)
		samp_1 = int((self.onset + self.fit_time_after ) / self.dt)
		t2 = np.arange(fit_t0, fit_t1 - 0.5/samprate, 1 / samprate)
		shifts = {st[0].stats.channel[2]:0, st[1].stats.channel[2]:distance, st[2].stats.channel[2]:-distance}
		colors = {'N':'r', 'E':'g', 'Z':'b'}
		plt.rcParams.update({'font.size': 22})
		plt.figure(figsize=(12, 6))
		for tr in st:
			ch = tr.stats.channel[2]
			print(len(t), len(tr.data), len(t2), len(self.mouse[samp_0:samp_1]))
			p = plt.plot(t, tr.data+shifts[ch], colors[ch], label = {'Z':'raw record','N':'','E':''}[ch])
			plt.text(xmin+(xmax-xmin)*0.05, shifts[ch]+distance*0.2, ch, color=colors[ch])
			if ch == 'E':
				m = plt.plot(t2, self.mouse[samp_0:samp_1]*self.amplitude*sin(self.phi)*cos(self.theta)+shifts[ch], colors[ch]+'--')
			elif ch == 'N':
				m = plt.plot(t2, self.mouse[samp_0:samp_1]*self.amplitude*cos(self.phi)*cos(self.theta)+shifts[ch], colors[ch]+'--')
			elif ch == 'Z':
				m = plt.plot(t2, self.mouse[samp_0:samp_1]*self.amplitude*sin(self.theta)+shifts[ch], colors[ch]+'--', label='simulated')
		plt.xlim(xmin, xmax)
		if legend:
			plt.legend(loc='upper right')
		plt.title(st[0].stats.network+':'+st[0].stats.station+', '+str(st[0].stats.starttime.date)+' '+str(st[0].stats.starttime.time)[:8])
		if ylabel:
			plt.ylabel(ylabel)
		plt.xlabel('time [s]')
		if yaxes:
			# plot y2 axis
			plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
			xmin,xmax,ymin,ymax = plt.axis()
			ax2=plt.twinx()
			const = self.mouse[int((self.nmax+self.onset/self.dt) / 2)]
			ax2.set_ylim(ymin/const,ymax/const)
			ax2.set_ylabel('acceleration [m/s$^2$]')
			ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		else:
			plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
		if outfile:
			plt.savefig(outfile, bbox_inches='tight')
		else:
			plt.show()



def NoiseTest1_demean(st, t, ratio=20):
	# Simple signal-to-noise criterion based on maximum-before-event / trace-maximum
	# Removes before-event-mean from entire record
	mx = []
	mn = []
	for i in range(3):
		tr = st[i]
		idx = int(t * tr.stats.sampling_rate) - 1
		mean = np.mean(tr.data[0:idx])
		tr.data = tr.data - mean
		mx.append(np.max(tr.data[0:idx]))
		mn.append(np.min(tr.data[0:idx]))
	m = st.max()
	m2 = max(max(m),abs(min(m)))
	m1 = max(max(mx), abs(min(mn)))
	if (m1*ratio > m2):
		return True

def demean(st, t):
	"""
	Calculates mean value from part of the records between starttime and the time ``t`` and removes this mean from entire record
	"""
	for i in range(3):
		tr = st[i]
		idx = int(t * tr.stats.sampling_rate) - 1
		mean = np.mean(tr.data[0:idx])
		tr.data = tr.data - mean

def NoiseTest2(st, t, ratio=8):
	# Simple signal-to-noise criterion based on maximum-before-event / trace-maximum
	t_start = UTCDateTime(st[0].stats.starttime)
	st_begin = st.slice(t_start, t_start+t)
	st_begin.detrend(type='linear')
	m = st.max()
	m2 = max(max(m),abs(min(m)))
	m = st_begin.max()
	m1 = max(max(m),abs(min(m)))
	if (m1*ratio > m2):
		return True

def ToDisplacement(st, bitrate=10):
	# Integrate and decimate
	st.integrate()
	if (st[0].stats.sampling_rate >= 2*bitrate):
		factor = int(st[0].stats.sampling_rate / bitrate)
		while (factor > 16):
			st.decimate(5)
			factor = int(factor / 5)
		st.decimate(factor)

def PrepareRecord(st, t, ratio_velocity=20, ratio_displacement=8, bitrate=10):
	if NoiseTest1_demean(st, t, ratio_velocity):
		return 'Record too noisy.'
	ToDisplacement(st, bitrate)
	if NoiseTest2(st, t, ratio_displacement):
		return 'Record too noisy after integration.'
