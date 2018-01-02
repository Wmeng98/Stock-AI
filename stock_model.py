# First add a few import statements for libraries we are going to use in this project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# pandas and numpy used to import data from csv sheet and manipulate it into correct arrays we want
# matplotlib used to plot data on graph for nice visualization of data we are working with
# tensorflow will build our computational graph and perform testing on it

# Shift + CTRL + Alt + J to select all occurences...
# store common variables to use at top in our stock prediction model
# these two csv variables will be feeded into functions


#
DJI_TRAIN_DATA = 'csv_files/DJI-LastYear.csv'
DJI_TEST_DATA = 'csv_files/DJI-LastMonth.csv'

# create some current training and testing to make project more scalable
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
stock = "Dow Jones Industrial"
trainSpan = "Year"


current_train_data = DJI_TRAIN_DATA
current_test_data = DJI_TEST_DATA
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# Easier way of doing things...

# Next we need to determine how many data points we want to test here

LEARNING_RATE = 0.1
NUM_EPOCHS = 100  # number of times we want to run our model

# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# recall Train data with Yearly historical and test with Monthly
# SHORTCUT : LineNumberEnd - 2
NUM_TRAIN_DATA_POINTS = 250  # Not 267 because we want diff on final price on one day and opening price on the FOLLOWING
# Also because you start counting nrows at zero
# Basically the xi is volume exchanged per day and tha[:-1] split is the configuration that is important!
NUM_TEST_DATA_POINTS = 18  # Not 23 because in our price diff calc won't be able to calc price diff for 23rd
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# ################ VARIABLES TO CHANGE ######################
# ############## Want to user input specific values ##############
userIn = -10  # default
while userIn != 8:
    print("*****************************")
    print("Predictive Stock Simulation: ")
    print("MAKE SURE TO FILL IN ALL PARAMETERS [1 - 7] to avoid program crash")
    print(" ")
    print("parameters (enter a value from [1 - 8] to update corresponding data)")
    print("Note: Invalid input of parameter values will cause the simulation to break")
    print("1.) stock name")
    print("2.) the duration of your training historical data (ex. last month, last year, last 10 years")
    print("3.) location of csv files (TRAINING) (ex. csv_files/DJI-LastYear.csv")
    print("4.) location of csv files (TESTING) (ex. csv_files/DJI-LastMonth.csv")
    print("5.) number of iterations on model simulation (default 100 times)")
    print("6.) Number of data points that need to be TRAINED (check in your training csv file)")
    print("7.) Number of data points that need to be TESTED (check in your testing csv file)")
    print("8.) finish and simulate")
    print(" ")
    print("*****************************")
    validCommands = [1,2,3,4,5,6,7,8]
    userIn = int(input("Enter command: "))
    if userIn in validCommands:
        if userIn == 1:
            stock = input("Enter stock name: ")
            print("UPDATED...")
        elif userIn == 2:
            trainSpan = input("Enter span of historical data in csv file for training purposes: ")
            print("UPDATED...")
        elif userIn == 3:
            current_train_data = input("Enter csv files (TRAINING) location: ")
            print(current_train_data)
            print("UPDATED...")
        elif userIn == 4:
            current_test_data = input("Enter csv files (TESTING) location: ")
            print("UPDATED...")
        elif userIn == 5:
            stringIteration = input("Enter number of model simulated iterations: ")
            NUM_EPOCHS = int(stringIteration)
            print("UPDATED...")
        elif userIn == 6:
            stringTrainData = input("Enter number of data points that need to be TRAINED: ")
            NUM_TRAIN_DATA_POINTS = int(stringTrainData)
            print("UPDATED...")
        elif userIn == 7:
            stringTestData = input("Enter number of data points that need to be TESTED: ")
            NUM_TEST_DATA_POINTS = int(stringTestData)
            print("UPDATED...")
        else:
            # userIn == 8 which means simulate
            continue
    else:
        print("Invalid Command")

print("")
print("")
print("Learning...")
print("")
print("")

# ################################################################

# Now want a function to load our data and convert it into the correct format and store it in some actual arrays

def load_stock_data(stock_name, num_data_points):  # function using panda library to read our csv sheet
    data = pd.read_csv(stock_name, skiprows=0, nrows=num_data_points, usecols=['Price', 'Open', 'Vol.'])

# panda function will return 3 separate arrays, one with final prices, open prices, and vol traded for the day
# skiprows skips first row (0)

    # We take each col, store in a respectable array, and then return a kind of tuple of the three arrays

    final_prices = data['Price'].astype(str).str.replace(',', '').astype(np.float)
    # returns final price array of prices in string format for each element and witout the ',', then converted intofloat
    # split, strip methods aren't working - Note

    opening_prices = data['Open'].astype(str).str.replace(',', '').astype(np.float)
    # same idea with open prices

    volumes = data['Vol.'].str.strip('MK').astype(np.float)

    # idea with volumes is different can strip any M(millions) or K(thousands) endings for elements in array
    # Then convert to a float

    # now all 3 arrays have been ocnverted to float arrays
    return final_prices, opening_prices, volumes
# ########################################################

# Idea calc difference between final price of one day and OPENING PRICES OF THE NEXT DAY
# Idea is to determine how the volume exchanged on one day is gonna affect the difference on prices in the next morning

# ####################

8# function to calc price difference

def calculate_price_differences(final_prices, opening_prices): #iterate through list and take the differences
    price_differences = []
    # array to store the float price differences

    for d_i in range(len(final_prices) - 1):

        price_difference = opening_prices[d_i + 1] - final_prices[d_i]
        # take diff between opening price of next day and final price of current day
        price_differences.append(price_difference)

    return price_differences

# will print out the three different float arrays

# finals, openings, volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
#
# print(calculate_price_differences(finals, openings))

# ##########################
# ##########################

# Accuracy test function
# Gives us the percent of correct guesses our model has made
# Idea is whwnever both values have the same sign, we want to increment our number of correct guesses
def calculate_accuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values[a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values[a_i]:
            num_correct += 1
    return (num_correct / len(actual_values)) * 100

# ##########################
# ##########################


# #####################################################
# Training data sets
train_final_prices, train_opening_prices, train_volumes = load_stock_data(current_train_data, NUM_TRAIN_DATA_POINTS)
# So now loaded stock data, now need to calc price diff
train_price_differences = calculate_price_differences(train_final_prices, train_opening_prices)
# Note how we're actually ignoring the final data point in the dataset because we cannot get the next day opening price
train_volumes = train_volumes[:-1]  # refer to ^^^
# #####################################################

# #####################################################
# Testing data sets (instead of calling upon curr training data and # training days we use - test data & test days
test_final_prices, test_opening_prices, test_volumes = load_stock_data(current_test_data, NUM_TEST_DATA_POINTS)
test_price_differences = calculate_price_differences(test_final_prices, test_opening_prices)
test_volumes = test_volumes[: -1]
# #####################################################


# ##############################################################
# Now have all the information we need to train and test our model, Now all that's left is to build our compu graph...

# We're gonna use tensorflow nodes and model after basic y = Wx + b

x = tf.placeholder(tf.float32, name='x') # act as a way to input data into our models for training/testing/eval purposes
# x is our list of volumes

W = tf.Variable([0.1], name='W')  # arbitrarily pick 0.1 for now, model aims to optimize this value however
# end up being one of the value we want optimize and train our W model
b = tf.Variable([0.1], name='b')

# linear regression equation stored as y
y = (W * x) + b  # The actual y value our computational model outputs

# we have now set up our initial computational graph shape, now need to build optimizer and loss functions/methods

y_predicted = tf.placeholder(tf.float32, name='y_predicted')  # this node stores what we expect our model to output
# kind of used as a check expect tensor, training purposes
# y_predicted is going to be our price differences...

# For simpliciy used reduced sum of the squares of elements in array as index marker for loss function

loss = tf.reduce_sum(tf.square(y - y_predicted))  # sum of all diff between expected y and actual y
# Also call upon an optimizer to optimize our loss value
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
# Adam Optimizer gave better results than Gradient Descent
# we'll be training optimizer by putting in y_predicted for tests and actual input x value selection to train
# think of volume as being the amount of x-axis the inputs
# ############################
# recap: Y = W * x + b
# Y is price diff between
# current and next day given volume, optimize W and b so that x (volumes each day) result in minimized price diff loss

# want loss which is y - y_predict to be as close to each other as possible, train W and b to give us optimized y which
# in a sense allows us to predict openings of next day for that stock
# ############################

# Create a session to run and train/test our linear regression model
session = tf.Session()
session.run(tf.global_variables_initializer())  # necessary for variable nodes
# the bigger the data set the greater the EPOCHS : general rule

# need feed in values along with optimizer such as x values and y_predicted values
# need final and opening prices, use our above helper functions
for _ in range(NUM_EPOCHS):  # want to TRAIN our Optimizer...
    session.run(optimizer, feed_dict={x: train_volumes, y_predicted: train_price_differences})
    # each of different price differences correspond with particular volume ...
    # As in final price of current day and current volume and opening price of next day

    # After model has been trained W and b have been modified to their optimal values, so we input x, get y outputs

# ############## Accuracy test ########################
results = session.run(y, feed_dict={x: test_volumes})
accuracy = calculate_accuracy(test_price_differences, results)  # expected vs actual(model determined) respectively
print("Accuracy of Model: {0:.2f}%".format(accuracy))

# ######## ADDITIONAL ACCURACY TESTING OPTION #################
# #############################################################

if accuracy <= 50:
    print("Accuracy of predictions is less than or equal to 50%, restart the program and try a different number of iterations to simulate the model with IN ORDER TO optimize predictions")

# #############################################################
# #############################################################


# ############## Accuracy test ########################
# keep in mind these are in thousands metrics, if have an M need proportionate accordingly



# We also want to be able to measure the accuracy of our model
# To do this: We feed in a bunch of EXPECTED values, and then compare them to the values our model is OUTPUTTING
#  and check how many our model got correct
# ########
# For our testing: We base correct or wrong off if predict and expect values are both positive or negative
# ########

# #####################################
# #####################################
# #####################################
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y, label="Optimized Linear Regression")
def maxVolume(arrayVolume): # at least one item in the array
    max = arrayVolume[0]
    byOneArray = arrayVolume[1:]
    for i in byOneArray:
        if i > max:
            max = i
    return max


# PLOTTING PURPOSES
plt.figure(1)
plt.plot(train_volumes, train_price_differences, 'bo', label="Training Data")
plt.title('Price Differences for Given Volumes For The Past '+trainSpan+'(s)')
plt.xlabel(stock + ' Exchange Volumes (Daily)')
plt.ylabel('Price Differences')
# Estimated linear equation
var = session.run([W, b])

varW = str(float(var[0]))
varb = str(float(var[1]))
varLinear = varW + '*x + ' + varb

rangeMax = maxVolume(train_volumes)
graph(varLinear, range(0, int(rangeMax) + 1))
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=2.0)

def graph_model():
    plt.show()

# #####################################
# #####################################
# #####################################

# Final input option to allow user to input volume exchange as x and predict
# y price diff of following day for the opening stock
print(" ")
print(" ")
print(" ")
print("Optimization complete, enter an x value for a valid volume exchanged on a current day to predict whether the stock will trade higher or lower on the following day")
print("Note: simulation accuracy  is optimized when volume exchange metrics (Million (M), Thousands (K)) are constant on the csv sheet")
print(" ")
print(" ")
userIn2 = -10
while userIn2 != 0:
    userIn2 = int(input("Enter [9] to view a graph of the linear regression model else [1] if volume metric is in thousands(K) else [2] for millions(M) else [0] to exit program: "))
    if userIn2 in [0, 1, 2, 9]:
        if userIn2 == 9:
            userIn2 = 0
            continue
        elif userIn2 == 1:
            volumeIn = int(input("Enter volume exchange (integer): "))
            y_diff = (var[0] * volumeIn) + var[1]
            print("Predicted difference in the price of the stock from current price to openeing price the next day is: " + str(y_diff))
            print(" ")
            print(" ")
        elif userIn2 == 2:
            volumeIn = int(input("Enter volume exchange (integer): "))
            y_diff = (var[0] * volumeIn) + var[1]
            print("Predicted difference in the price of the stock from current price to openeing price the next day is: " + str(y_diff))
            print(" ")
            print(" ")
        else:  # userIn2 == 0
            continue
    else:  # userIn2 == 0
        print("Invalid Command")
        print(" ")
        print(" ")

graph_model()
print(" ")
print("Simulation Terminated...")