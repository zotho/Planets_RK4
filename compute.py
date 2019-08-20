#!/usr/bin/env python

import numpy as np

N_DIM: int = 3


class Space:
	"""Space, containing planets.

	planets:	[x, ..., 	vx, ..., 	m]

	matrix:	   [[x1, ...,	x1, ..., 	dx1, ..., 	dist1],
				[x1, ..., 	x2, ...,	dx2, ...,	dist2],
				...,									   ]

	"""
	def __init__(self):
		self.planets: np.ndarray = np.zeros([1, N_DIM])
		self.null_planet: int = 0

	def add_planet(self, vector: np.ndarray):
		rows, cols = self.planets.shape
		if self.null_planet >= rows:
			new_array = np.zeros([rows + 1, cols])
			new_array[:rows, :] = self.planets
			self.planets = new_array
		self.planets[self.null_planet, ] = vector
		self.null_planet += 1

	def update(self, time: np.float = 1):
		rows, cols = self.planets.shape
		matrix = np.zeros((rows * rows, cols * N_DIM + 1))
		matrix[:, :cols * 2] = self.concatenate_per_row(self.planets, self.planets)
		matrix[:, cols * 2: cols * 3] = matrix[:, cols:cols * 2] - matrix[:, :cols]
		matrix[:, cols * 3:] = np.sqrt(np.sum(matrix[:, cols * 2: cols * 3] ** 2, axis=1).reshape((rows ** 2, 1)))
		print(matrix[:, :])

	@staticmethod
	def concatenate_per_row(a: np.ndarray, b: np.ndarray):
		m1, n1 = a.shape
		m2, n2 = b.shape
		out = np.zeros((m1, m2, n1 + n2), dtype=a.dtype)
		out[:, :, :n1] = a[:, None, :]
		out[:, :, n1:] = b
		return out.reshape(m1 * m2, -1)


if __name__ == "__main__":
	print("Starting")
	np.set_printoptions(precision=3)
	s = Space()
	s.add_planet(np.array([1, 2, 3]))
	s.add_planet(np.array([4, 5, 6]))
	s.add_planet(np.array([4, 5, 7]))
	print(s.planets)
	s.update()
