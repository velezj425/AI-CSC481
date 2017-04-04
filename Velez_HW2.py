# Julian Velez
# 3/29/2017
# CSC 481 : Artificial Intelligence
# Homework 2 : Gradient Descent

# hypothesized value function
def hyp(w_0, w_1, x):
    hw = w_1 * x + w_0
    return hw

# batch gradient descent function
def batch_desc(x, y, w_0, w_1, alpha):
    # define our variables
    itr = 0
    w0prime = 0
    w1prime = 0
    cont = True

    while cont:
        itr += 1
        # find our hypothesized values
        hw_list = []
        for i in range(0, len(x) - 1):
            hw = hyp(w_0, w_1, x[i])
            hw_list.append(hw)
    
        # find difference values
        delta_0 = 0.0
        delta_1 = 0.0
        for j in range(0, len(y) - 1):
            diff = y[j] - hw_list[j]
            delta_0 += diff
            delta_1 += (diff * x[j])
        
        # find w0' and w1'
        w0prime = w_0 + alpha * delta_0
        w1prime = w_1 + alpha * delta_1
        
        # check for convergence
        if (abs(w_0 - w0prime) < pow(10, -10) and (abs(w_1 - w1prime) < pow(10, -10))) or itr >= pow(10, 6):
            fin_val = [w0prime, w1prime]
            print("Iterations: " + str(itr))
            cont = False
        else:
            w_0 = w0prime
            w_1 = w1prime
    
    return fin_val

# stochastic gradient descent function
def stoc_desc(x, y, w_0, w_1, alpha):
    itr = 0
    cont = True

    while cont:
        for i in range(0, len(x)):
            w0prime = w_0 + alpha * (y[i] - hyp(w_0, w_1, x[i]))
            w1prime = w_1 + alpha * (y[i] - hyp(w_0, w_1, x[i])) * x[i]
            itr += 1

        # check for convergence
        if (abs(w_0 - w0prime) < pow(10, -10) and (abs(w_1 - w1prime) < pow(10, -10))) or itr >= pow(10, 6):
            fin_val = [w0prime, w1prime]
            print("Iterations: " + str(itr))
            cont = False
        else:
            w_0 = w0prime
            w_1 = w1prime
    
    return fin_val


# main function
def main():
    # define variables
    w_0 = 0.1
    w_1 = 0.2
    alpha = 0.0001
    x = [2, 4, 6, 7, 8, 10]
    y = [5, 7, 14, 14, 17, 19]

    # perform batch descent
    print("Batch Gradient Descent: ")
    val = batch_desc(x, y, w_0, w_1, alpha)
    print("hw(x) = " + str(val[1]) + "x + " + str(val[0]))
    print("i. x = 5")
    print("y = " + str(val[1] * 5 + val[0]))
    print("ii. x = -100")
    print("y = " + str(val[1] * -100 + val[0]))
    print("iii. x = 100")
    print("y = " + str(val[1] * 100 + val[0]))

    # perform stochastic descent
    print("\nStochastic Gradient Descent: ")
    val = stoc_desc(x, y, w_0, w_1, alpha)
    print("hw(x) = " + str(val[1]) + "x + " + str(val[0]))
    print("i. x = 5")
    print("y = " + str(val[1] * 5 + val[0]))
    print("ii. x = -100")
    print("y = " + str(val[1] * -100 + val[0]))
    print("iii. x = 100")
    print("y = " + str(val[1] * 100 + val[0]))

main()
