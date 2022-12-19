import math
import sys
import os
import random
from PIL import Image
from PIL import ImageOps
import numpy as np
import libs.ssim as ssim
import libs.utils as ut
import ctls.mpc as mpccontroller
import ctls.random as randomcontroller
import ctls.bangbang as bangbangcontroller
import ctls.dct as dctController

def image_to_matrix(path):
	img = Image.open(str(path))
	img = ImageOps.grayscale(img)
	img_data = img.getdata()
	img_tab = np.array(img_data)
	w,h = img.size
	img_mat = np.reshape(img_tab, (h,w))
	return img_mat

def compute_ssim(path_a, path_b):
	matrix_a = image_to_matrix(path_a)
	matrix_b = image_to_matrix(path_b)
	return ssim.compute_ssim(matrix_a, matrix_b)
	
def generate_random_configuration():
	# random quality - min or max
	if bool(random.getrandbits(1)):
		quality = 100
	else:
		quality = 1
	# random sharpen - min or max
	if bool(random.getrandbits(1)):
		sharpen = 5
	else:
		sharpen = 0
	# random noise - min or max
	if bool(random.getrandbits(1)):
		noise = 5
	else:
		noise = 0
	# return random choice
	return (quality, sharpen, noise)

def pathIn(frame_in):
	framename = str(1).zfill(8) + '.jpg'
	img_in = frame_in + '/' + framename

	return img_in

def pathOut(frame_out):
	framename = str(1).zfill(8) + '.jpg'
	img_out = frame_out + '/' + framename
	return img_out

def encode(i, frame_in, frame_out, quality, sharpen, noise):
	img_in = pathIn(frame_in)
	img_out = pathOut(frame_out)
	# generating os command for conversion
	# sharpen actuator
	if sharpen != 0:
		sharpenstring = ' -sharpen ' + str(sharpen) + ' '
	else:
		sharpenstring = ' '
	# noise actuator
	if noise != 0:
		noisestring = ' -noise ' + str(noise) + ' '
	else:
		noisestring = ' '
	# command setup
	command = 'convert {file_in} -quality {quality} '.format(
			file_in = img_in, quality = quality)
	command += sharpenstring
	command += noisestring
	command += img_out
	# executing conversion
	os.system(command)
	# computing current values of indices
	current_quality = compute_ssim(img_in, img_out)
	current_size = os.path.getsize(img_out)
	return (current_quality, current_size)

# -------------------------------------------------------------------

def main(args):

	# parsing arguments
	mode = args[1] # identify, mpc
	folder_frame_in = args[2]
	folder_frame_out = args[3]
	folder_results = args[4]
	setpoint_quality = float(args[5])
	setpoint_compression = float(args[6])
	mode = 'dct'
	# getting frames and opening result file
	path, dirs, files = os.walk(folder_frame_in).next()
	frame_count = len(files)
	final_frame = frame_count + 1
	log = open(folder_results + '/results.csv', 'w')
	# print(frame_count);
	if mode == "dct":
		controller = dctController.DCTController()
		controller = mpccontroller.initialize_mpc()
	elif mode == "random":
		controller = randomcontroller.RandomController()
	elif mode == "bangbang":
		controller = bangbangcontroller.BangbangController()
	elif mode == "dct":
		controller = dctController.DCTController()
	
	# initial values for actuators
	ctl = np.matrix([[100], [0], [0]])
	for i in range(1, final_frame):
		# main loop
		quality = np.round(ctl.item(0))
		sharpen = np.round(ctl.item(1))
		noise = np.round(ctl.item(2))

		# encoding the current frame
		(current_quality, current_size) = \
			encode(i, folder_frame_in, folder_frame_out, quality, sharpen, noise)
		log_line = '{i}, {quality}, {sharpen}, {noise}, {ssim}, {size}'.format(
			i = i, quality = quality, sharpen = sharpen, noise = noise,
			ssim = current_quality, size = current_size)
		print >>log, log_line
		
		setpoints = np.matrix([[setpoint_quality], [setpoint_compression]])
		current_outputs = np.matrix([[current_quality], [current_size]])
		
		# computing actuator values for the next frame
		if mode == "mpc":
			try:
				ctl = controller.compute_u(current_outputs, setpoints)
			except Exception:
				pass
    
		elif mode == "random":
			ctl = controller.compute_u()
			
		elif mode == "bangbang":
			ctl = controller.compute_u(current_outputs, setpoints)

		elif mode == "dct":
			matrix_a = image_to_matrix(pathIn(folder_frame_in))
			ctl = controller.do_compress(matrix_a, current_outputs, setpoints)
			
	print(" done")

if __name__ == "__main__":
	main(sys.argv)
  
