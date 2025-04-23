
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np



dir_path = r"biodcase_development_set\train\audio"
save_path = r"biodcase_development_set\train\spectrogrammes"

def convert_wav_to_float(data):
	if data.dtype == np.uint8:
		data = (data - 128) / 128.
	elif data.dtype == np.int16:
		data = data / 32768.
	elif data.dtype == np.int32:
		data = data / 2147483648.
	return data


for dir in os.listdir(dir_path):
	print("dir =", dir)
	path = dir_path + r"\\" + dir 
	temp_save_dir = save_path + "\ "[:-1] + dir
	print("temp_save =", temp_save_dir)
	for file in os.listdir(path):
		if not os.path.exists(temp_save_dir):
			os.makedirs(temp_save_dir)
		temp_save_path = temp_save_dir + "\ "[:-1] + file[:-4] + ".png"

		if not os.path.exists(temp_save_path):
			file = path + r"\\" + file
			sampling_frequency, wav_data = wavfile.read(file)
			n_samples = len(wav_data)
			total_duration = n_samples / sampling_frequency
			sample_times = np.linspace(0, total_duration, n_samples)


			
			plt.plot(sample_times, wav_data, color="k");
			print("Sauvegardé au :", temp_save_path)
			plt.savefig(temp_save_path)
			plt.close()  # Completely closes the current figure to avoid ressource accumulation 
			
