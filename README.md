# Mushroom_Problem
Kaggle's mushroom problem

Edible :) 😋

  or
  
Poisnous !! 😨


## How to run

### Windows
- clone the repo on your local using `git clone git@github.com:sourabhmehtasg/Mushroom_Problem.git`.
- Open the command prompt, `cd path\to\Mushroom_Problem\App`
- type : `init.bat`, this will up and run the server on ` http://127.0.0.1:5000/`
- Open a browser and go to: `http://127.0.0.1:5000/showInput`, this is the main page where user can select various properties of the mushroom he/she wants to classify.
- Click the button on the UI and the result will be displayed below the button.


## Model Trainign
There is separate "ml_model" folder containing jupyter notebook named "Model_Training" for data exploration and model training.

## The Classifier model
The final model used is the trained using XGBoost algorithm and saved as "xgb_model_1.pickle.dat".

## Result
The classification result for the particular user input can be viewed on the same page under the "Edible or Poisnous" button.

It will give result as: Edible :) 😋 or Poisnous !! 😨
