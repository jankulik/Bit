# Machinery price prediction tool

In order to run the code, make sure to install the following libraries: 
```bash
pip install tensorflow
pip install pandas
pip install seaborn
pip install matplotlib
```

Code for preprocessing, modelling, testing and visualisation is available in the `main.ipynb` file. This file also contains explanation about preprocessing and model architecture.

In order to make predictions with the model, `predict.py` file can be used. It takes `data/input_data.csv` file as input and produces `data/output_data.csv` file as output. The input should be of the same structure as the original `BIT_AI_assignment_data.csv` data file, except the `Sales Price` column can be empty. Format of the output is exactly the same, except the `Sales Price` column contains predicted values.