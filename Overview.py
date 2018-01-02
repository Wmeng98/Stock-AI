# This is a Python programto model and predict whether a stock price will increase or decrease the very next trading day
# Most beneficial for day trading simulations and strategies
#
# Model prediction will bebased on the amount of volume exchanged by the end of a certain day (only) no other parameters
# have been taken into consideration yet (such as global and current events)
#
# Logic
#     - If the price of the stock at the end of the day is different than at the beginning of the next day. We want to
#        retrieve this difference and figure out if the stock price will go up or down based on previous data
#     - If model predictsthat the stock price will go up, then instruct user to buy at the end of the day and sell right
#         on the very next trading day
#     - Considerations: accuracy of model will be around 60% accurate resulting in a net gain over the long run
#
# Getting Started
#     - Start by exploring the data sets (using investing dot com)
#     - Then look at how to import, format, and manipulate the retrieved data
#     -Build/Train computational model
#     -Test accuracy of result
#     - Goal: Program willtake in volume exchanged for a given day and predict whether or not the stock price will ++ or
#         -- the NEXT MORNING as a result!


# INSTALLATION NOTES OF CONSIDERATION:
# ####################################
# 1) Make sure you're using python3.6 x 64-bit based
# 2) Make sure to install TensorFlow
# 3) Make sure you check off inherit all global variables if  in pycharm IDE for interpretor purposes if import
#     tensforflow error occurs in your program
# 4) Install pandas and matplotlib libraries into your py project if you haven't already
#
# GOOD NOTES WHEN RUNNING DATA IN PROGRAM
# 1) Make sure imported csv stock data has a column for volume of exchanges
# 2) Program works best when all volume exhanges are in Millions(M) or thousands(K) (Ideally: mutually exclusive)



