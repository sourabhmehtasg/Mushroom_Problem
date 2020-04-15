import flask
from flask import Flask, render_template, request, json
import pandas as pd
# from sklearn.preprocessing import LabelEncoder
import sklearn
import pickle
import xgboost
import os
import logging
import pathlib
import os
import inspect
import sys
from datetime import datetime


# setting logger -
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
log_dir_path = os.path.join(curr_dir,'log')
log_dt_now = datetime.now().date().strftime("%d-%m-%Y")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler(log_dir_path+'/'+log_dt_now+'_MushroomClassifier.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# logger setting ends -


app = Flask(__name__)

# model_loc = ""
# model_dir = os.path.join(model_loc, "ml_model")
# print("-==-=*"*20,model_dir)
# loaded_model = pickle.load(open(model_dir+"\\xgb_model_1.pickle.dat", "rb"))
loaded_model = pickle.load(open("ml_model\\xgb_model_1.pickle.dat", "rb"))

col_ls = ['bruises',
 'gill-attachment',
 'gill-spacing',
 'gill-size',
 'stalk-shape',
 'cap-shape_b',
 'cap-shape_c',
 'cap-shape_f',
 'cap-shape_k',
 'cap-shape_s',
 'cap-shape_x',
 'cap-surface_f',
 'cap-surface_g',
 'cap-surface_s',
 'cap-surface_y',
 'cap-color_b',
 'cap-color_c',
 'cap-color_e',
 'cap-color_g',
 'cap-color_n',
 'cap-color_p',
 'cap-color_r',
 'cap-color_u',
 'cap-color_w',
 'cap-color_y',
 'odor_a',
 'odor_c',
 'odor_f',
 'odor_l',
 'odor_m',
 'odor_n',
 'odor_p',
 'odor_s',
 'odor_y',
 'gill-color_b',
 'gill-color_e',
 'gill-color_g',
 'gill-color_h',
 'gill-color_k',
 'gill-color_n',
 'gill-color_o',
 'gill-color_p',
 'gill-color_r',
 'gill-color_u',
 'gill-color_w',
 'gill-color_y',
 'stalk-surface-above-ring_f',
 'stalk-surface-above-ring_k',
 'stalk-surface-above-ring_s',
 'stalk-surface-above-ring_y',
 'stalk-surface-below-ring_f',
 'stalk-surface-below-ring_k',
 'stalk-surface-below-ring_s',
 'stalk-surface-below-ring_y',
 'stalk-color-above-ring_b',
 'stalk-color-above-ring_c',
 'stalk-color-above-ring_e',
 'stalk-color-above-ring_g',
 'stalk-color-above-ring_n',
 'stalk-color-above-ring_o',
 'stalk-color-above-ring_p',
 'stalk-color-above-ring_w',
 'stalk-color-above-ring_y',
 'stalk-color-below-ring_b',
 'stalk-color-below-ring_c',
 'stalk-color-below-ring_e',
 'stalk-color-below-ring_g',
 'stalk-color-below-ring_n',
 'stalk-color-below-ring_o',
 'stalk-color-below-ring_p',
 'stalk-color-below-ring_w',
 'stalk-color-below-ring_y',
 'veil-color_n',
 'veil-color_o',
 'veil-color_w',
 'veil-color_y',
 'ring-number_n',
 'ring-number_o',
 'ring-number_t',
 'ring-type_e',
 'ring-type_f',
 'ring-type_l',
 'ring-type_n',
 'ring-type_p',
 'spore-print-color_b',
 'spore-print-color_h',
 'spore-print-color_k',
 'spore-print-color_n',
 'spore-print-color_o',
 'spore-print-color_r',
 'spore-print-color_u',
 'spore-print-color_w',
 'spore-print-color_y',
 'population_a',
 'population_c',
 'population_n',
 'population_s',
 'population_v',
 'population_y',
 'habitat_d',
 'habitat_g',
 'habitat_l',
 'habitat_m',
 'habitat_p',
 'habitat_u',
 'habitat_w']


def one_hot_decoder(v, option, ls):
	df = pd.DataFrame()
	v_n = v+"_"+option
	for i in ls:
		if v_n == i:
			df[i] = [1]
		else:
			df[i] = [0]

	return df


def feature_creator(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat):
	# feature_df = pd.DataFrame(columns=col_ls)
	feature_df = pd.DataFrame()
	# taking on binary variables:
	ls = pd.Series()
	if bruises == "t":
		feature_df["bruises"] = [1]
	else:
		feature_df["bruises"] = [0]

	if gill_attachment == "f":
		feature_df["gill-attachment"] = [1]
	else:
		feature_df["gill-attachment"] = [0]

	if gill_spacing == "w":
		feature_df["gill-spacing"] = [1]
	else:
		feature_df["gill-spacing"] = [0]

	if gill_size == "n":
		feature_df["gill-size"] = [1]
	else:
		feature_df["gill-size"] = [0]

	if stalk_shape == "t":
		feature_df["stalk-shape"] = [1]
	else:
		feature_df["stalk-shape"] = [0]

	cap_shape_df = one_hot_decoder("cap-shape", cap_shape, ['cap-shape_b','cap-shape_c','cap-shape_f','cap-shape_k',  'cap-shape_s',  'cap-shape_x'])
	feature_df = feature_df.join(cap_shape_df)

	cap_surface_df = one_hot_decoder("cap_surface", cap_surface, ['cap-surface_f', 'cap-surface_g', 'cap-surface_s', 'cap-surface_y'])
	feature_df = feature_df.join(cap_surface_df)

	cap_color_df = one_hot_decoder("cap-color", cap_color, ['cap-color_b', 'cap-color_c', 'cap-color_e', 'cap-color_g', 'cap-color_n', 'cap-color_p', 'cap-color_r', 'cap-color_u', 'cap-color_w', 'cap-color_y'])
	feature_df = feature_df.join(cap_color_df)

	odor_df = one_hot_decoder("odor", odor, ['odor_a', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p', 'odor_s', 'odor_y'])
	feature_df = feature_df.join(odor_df)

	gill_color_df = one_hot_decoder("gill-color", gill_color, ['gill-color_b', 'gill-color_e', 'gill-color_g', 'gill-color_h', 'gill-color_k', 'gill-color_n', 'gill-color_o', 'gill-color_p', 'gill-color_r', 'gill-color_u', 'gill-color_w', 'gill-color_y'])
	feature_df = feature_df.join(gill_color_df)

	stalk_surface_above_ring_df = one_hot_decoder("stalk-surface-above-ring", stalk_surface_above_ring, ['stalk-surface-above-ring_f', 'stalk-surface-above-ring_k', 'stalk-surface-above-ring_s', 'stalk-surface-above-ring_y'])
	feature_df = feature_df.join(stalk_surface_above_ring_df)

	stalk_surface_below_ring_df = one_hot_decoder("stalk-surface-below-ring", stalk_surface_below_ring, ['stalk-surface-below-ring_f', 'stalk-surface-below-ring_k', 'stalk-surface-below-ring_s', 'stalk-surface-below-ring_y'])
	feature_df = feature_df.join(stalk_surface_below_ring_df)

	stalk_color_above_ring_df = one_hot_decoder("stalk-color-above-ring", stalk_color_above_ring, ['stalk-color-above-ring_b', 'stalk-color-above-ring_c', 'stalk-color-above-ring_e', 'stalk-color-above-ring_g', 'stalk-color-above-ring_n', 'stalk-color-above-ring_o', 'stalk-color-above-ring_p', 'stalk-color-above-ring_w', 'stalk-color-above-ring_y'])
	feature_df = feature_df.join(stalk_color_above_ring_df)

	stalk_color_below_ring_df = one_hot_decoder("stalk-color-below-ring", stalk_color_below_ring, ['stalk-color-below-ring_b', 'stalk-color-below-ring_c', 'stalk-color-below-ring_e', 'stalk-color-below-ring_g', 'stalk-color-below-ring_n', 'stalk-color-below-ring_o', 'stalk-color-below-ring_p', 'stalk-color-below-ring_w', 'stalk-color-below-ring_y'])
	feature_df = feature_df.join(stalk_color_below_ring_df)

	veil_color_df = one_hot_decoder("veil-color", veil_color, ['veil-color_n', 'veil-color_o', 'veil-color_w', 'veil-color_y'])
	feature_df = feature_df.join(veil_color_df)

	ring_number_df = one_hot_decoder("ring-number", ring_number, ['ring-number_n', 'ring-number_o', 'ring-number_t'])
	feature_df = feature_df.join(ring_number_df)

	ring_type_df = one_hot_decoder("ring-type", ring_type, ['ring-type_e', 'ring-type_f', 'ring-type_l', 'ring-type_n', 'ring-type_p'])
	feature_df = feature_df.join(ring_type_df)

	spore_print_color_df = one_hot_decoder("spore-print-color", spore_print_color, ['spore-print-color_b', 'spore-print-color_h', 'spore-print-color_k', 'spore-print-color_n', 'spore-print-color_o', 'spore-print-color_r', 'spore-print-color_u', 'spore-print-color_w', 'spore-print-color_y'])
	feature_df = feature_df.join(spore_print_color_df)

	population_df = one_hot_decoder("population", population, ['population_a', 'population_c', 'population_n', 'population_s', 'population_v', 'population_y'])
	feature_df = feature_df.join(population_df)

	habitat_df = one_hot_decoder("habitat", habitat, ['habitat_d', 'habitat_g', 'habitat_l', 'habitat_m', 'habitat_p', 'habitat_u', 'habitat_w'])
	feature_df = feature_df.join(habitat_df)

	return feature_df


@app.route('/showInput')
def showInputFields():
    return render_template('input.html')


@app.route('/classify',methods=['POST'])
def MushroomClassifier():
	data = request.get_json()
	logger.info(":::::::::----------------"+ str(data))
    # read the posted values from the UI
	cap_shape = data['cap_shape']
	cap_surface = data['cap_surface']
	cap_color = data['cap_color']
	bruises = data['bruises']
	odor = data['odor']
	gill_attachment = data['gill_attachment']
	gill_spacing = data['gill_spacing']
	gill_size = data['gill_size']
	gill_color = data['gill_color']
	stalk_shape = data['stalk_shape']
	# stalk_root = data['stalk_root']
	stalk_surface_above_ring = data['stalk_surface_above_ring']
	stalk_surface_below_ring = data['stalk_surface_below_ring']
	stalk_color_above_ring = data['stalk_color_above_ring']
	stalk_color_below_ring = data['stalk_color_below_ring']
	# veil_type = data['veil_type']
	veil_color = data['veil_color']
	ring_number = data['ring_number']
	ring_type = data['ring_type']
	spore_print_color = data['spore_print_color']
	population = data['population']
	habitat = data['habitat']
 
    # validate the received values
	# print(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat)   
	logger.info(cap_shape+ cap_surface+ cap_color+ bruises+ odor+ gill_attachment+ gill_spacing+ gill_size+ gill_color+ stalk_shape+ stalk_surface_above_ring+ stalk_surface_below_ring+ stalk_color_above_ring+ stalk_color_below_ring+ veil_color+ ring_number+ ring_type+ spore_print_color+ population+ habitat)   
	feature_vector_df = feature_creator(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_color, ring_number, ring_type, spore_print_color, population, habitat)
	# loaded_model = pickle.load(open("D:\\Mushroom_Problem\\ml_model\\xgb_model_1.pickle.dat", "rb"))

	print(feature_vector_df)
	pred = loaded_model.predict(feature_vector_df)[0]
	if pred == 1:
		logger.info("Prediction is:= "+ str(pred)+": Poisonous !!")
	else:
		logger.info("Prediction is:= "+ str(pred)+": Edible :)")

	if pred == 1:
		return flask.jsonify({'html':'<h3 class="jumbotron" align="center"> Poisonous !! <span style="font-size:100px">&#128552;</span></h3>'})
	else:
		return flask.jsonify({'html':'<h3 class="jumbotron" align="center"> Edible :) <span style="font-size:100px">&#128523;</span></h3>'})


if __name__ == "__main__":
	# model_loc = sys.argv[0]
	app.run()