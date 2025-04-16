
import os
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pandas as pd
from datetime import datetime


dir_path = r"biodcase_development_set\train\spectrogrammes"
annotation_path = r"biodcase_development_set\train\annotations"
save_path = r"biodcase_development_set\train\imagettes"

if not os.path.exists(save_path):
  os.makedirs(save_path)

def crop_image(pathname,x_start,x_end,y_low,y_high):
  #To do
  return 
  
def load_csv(dir) :
  df = pd.read_csv(dir)
  return df

def get_params(ligne):
  low = float(ligne['low_frequency'])
  high = float(ligne['high_frequency'])
  start = pd.to_datetime(ligne['start_datetime'])
  end = pd.to_datetime(ligne['end_datetime'])
  return (start,end,low,high)


for dir in os.listdir(dir_path):
  print("dir =", dir)
  path = dir_path + "\ "[:-1] + dir 
  temp_save_dir = save_path + "\ "[:-1] + dir
  print("temp_save =", temp_save_dir)
  
  temp_annotation_file = annotation_path + "\ "[:-1] + dir + ".csv"
  csv = load_csv(temp_annotation_file)
  print("Le csv ", dir ," a été chargé avec succès.")
  c = 1
  for file in os.listdir(path):

    if not os.path.exists(temp_save_dir):
      os.makedirs(temp_save_dir)
    temp_save_path = temp_save_dir + "\ "[:-1] + file[:-4] + ".png"
    

    if not os.path.exists(temp_save_path):
      ligne = csv.iloc[c]
      (start,end,low,high) = get_params(ligne)
      c+=1
      file = path + r"\\" + file

      img = crop_image(file,start,end,low,high)
      plt.savefig(temp_save_path)
      print("Sauvegardé au :", temp_save_path)
      plt.close()  # Completely closes the current figure to avoid ressource accumulation 
      
