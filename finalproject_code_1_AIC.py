import numpy as np
import scipy.stats as stats


def functionA(x, p1):
    return stats.halfnorm.logpdf(x, loc=0, scale=p1)


def functionC(x, p1, p2):
    return stats.levy.logpdf(x, loc=p1, scale=p2)


def functionE(x, p1, p2):
    return stats.pareto.logpdf(x, p1, loc=0, scale=p2)


def functionF(x, p1, p2):
    return stats.gamma.logpdf(x, p1, loc=0, scale=p2)


def functionH(x, p1, p2):
    return stats.cauchy.logpdf(x, loc=p1, scale=p2)


def objective_one(model_function, data, p1):
    result = 0
    for i in data:
        result += model_function(i, p1)
    return result


def objective_two(model_function, data, p1, p2):
    result = 0
    for i in data:
        result += model_function(i, p1, p2)
    return result


def simulated_annealing_one(model_function, data, cooling_rate, n_iter):
    T = 1
    def cool(T): return cooling_rate * T

    curr_x = 1
    curr_obj = objective_one(model_function, data, curr_x)
    best_x = curr_x
    best_obj = curr_obj
    for i in range(n_iter):
        T = cool(T)  # decrease T as scheduled
        new_x = curr_x + np.random.uniform(-1, 1)  # search neighbourhood
        new_obj = objective_one(model_function, data, new_x)

        if ((new_obj > curr_obj) or (np.random.rand()
                                     # if likelihood is higher or T is high enough
                                     < np.exp((new_obj - curr_obj) / T))):
            curr_x = new_x
            curr_obj = new_obj
            T = 1

        if (best_obj < curr_obj):  # if likelihood is best
            best_x = curr_x
            best_obj = curr_obj

    return best_x, best_obj


def simulated_annealing_two(model_function, data, cooling_rate, n_iter):
    T = 1
    def cool(T): return cooling_rate * T

    curr_x = 1
    curr_y = 1

    curr_obj = objective_two(model_function, data, curr_x, curr_y)

    best_x = curr_x
    best_y = curr_y
    best_obj = curr_obj
    for i in range(n_iter):
        T = cool(T)  # decrease T as scheduled
        new_x = curr_x + np.random.uniform(-0.5, 0.5)  # search neighbourhood
        new_y = curr_y + np.random.uniform(-0.5, 0.5)
        new_obj = objective_two(model_function, data, new_x, new_y)

        # if likelihood is higher or T is high enough
        if ((new_obj > curr_obj) or (np.random.rand() < np.exp((new_obj - curr_obj) / T))):
            curr_x = new_x
            curr_y = new_y
            curr_obj = new_obj
            T = 1

        if (best_obj < curr_obj):  # if likelihood is best
            best_x = curr_x
            best_y = curr_y
            best_obj = curr_obj

    return [best_x, best_y], best_obj


# settings for annealing
n_iter = 1000
cooling_rate = 0.9
data1 = np.genfromtxt("data1.csv", delimiter=",")
data2 = np.genfromtxt("data2.csv", delimiter=",")
data3 = np.genfromtxt("data3.csv", delimiter=",")
dataset = [data1, data2, data3]
modelset1 = [functionA, functionF]  # only survived models are used
modelset2 = [functionC, functionE]
modelset3 = [functionH]
modelsets = [modelset1, modelset2, modelset3]
for data in dataset:
    idx = dataset.index(data)
    models = modelsets[idx]
    for model in models:
        if (model == functionA):  # when it has one parameter
            best_index, LMLE_value = simulated_annealing_one(
                model, data, cooling_rate, n_iter)
            print("best p1: {}, best log-MLE: {}".format(
                round(best_index, 4), round(LMLE_value, 4)))
            # calculate AIC
            print("AIC: {}".format(round(-2*LMLE_value + 2*1, 4)))
        else:  # when it has two parameters
            best_indexs, LMLE_value = simulated_annealing_two(
                model, data, cooling_rate, n_iter)
            print("best p1: {}, best p2: {}, log-MLE: {}".format(
                round(best_indexs[0], 4), round(best_indexs[1], 4), round(LMLE_value, 4)))
            # calculate AIC
            print("AIC: {}".format(round(-2*LMLE_value + 2*2, 4)))
    print("-------------")
