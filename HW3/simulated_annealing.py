import numpy as np
import copy, random
import matplotlib.pyplot as plt

#number of items
N = 100

#weight limit
W = 3000

# weights of items
w = np.array([430, 763, 965, 848, 481, 336, 346, 381, 252, 218, 234, 906, 398,
       749, 343, 281, 773, 589, 896, 342, 640, 935, 876, 456, 629, 296,
       515, 946, 843, 554, 655, 906, 528, 273, 477, 347, 495, 294, 638,
       552, 607, 815, 394, 608, 694, 905, 826, 528, 739, 628, 810, 709,
       822, 296, 657, 654, 653, 892, 410, 812, 780, 469, 837, 450, 417,
       688, 308, 345, 616, 904, 827, 453, 690, 782, 626, 752, 994, 236,
       924, 303, 881, 617, 241, 559, 337, 930, 352, 922, 377, 865, 825,
       536, 499, 729, 359, 897, 959, 293, 658, 983])

# values of items
v = np.array([25, 27, 15, 25, 13, 15, 18, 24, 25, 30, 12, 18, 28, 30, 20, 26, 24,
       30, 14, 20, 15, 30, 18, 26, 16, 24, 11, 16, 14, 13, 13, 14, 30, 12,
       21, 13, 12, 28, 22, 14, 10, 20, 29, 19, 30, 16, 12, 24, 28, 27, 29,
       18, 16, 27, 30, 29, 17, 19, 26, 12, 24, 15, 27, 16, 15, 15, 19, 14,
       22, 30, 19, 30, 19, 24, 27, 16, 12, 27, 24, 17, 12, 18, 11, 14, 27,
       13, 23, 11, 26, 22, 12, 13, 15, 20, 20, 24, 12, 10, 14, 13])

def simulated_annealing():
    # YOUR CODE HERE
    # return a trace of values resulting from your simulated annealing
    
    def weight(bag):
        weight = 0
        for index in bag:
            weight = weight + w[index]
        return weight
        
    def value(bag):
        value = 0
        for index in bag:
            value = value + v[index]
        return value
    
    vals = []
    for i in range(10):
        bag = w.argsort()[:10].tolist()
        for j in range(5000):            
            remove = np.random.choice(bag)
            append = np.random.choice([d for d in range(N) if d not in bag])
            new_bag = copy.deepcopy(bag)
            new_bag.remove(remove)
            new_bag.append(append)
            
            if weight(new_bag) <= W:
                if (value(new_bag) >= value(bag)) or (random.random < .01/(j+1) + .01/(value(bag) - value(new_bag) + 0.001)):
                    bag = copy.deepcopy(new_bag)
        print(weight(bag))    
        vals.append(value(bag))
            
    return vals

if __name__ == "__main__":
    # Greedy result is maximize v/w
    vw_ratio = sorted(map(lambda x: (x, 1.*v[x]/w[x]), range(N)), key= lambda x: -x[1])
    greedy_val = 0
    greedy_weight = 0
    greedy_bag = []
    index = 0
    while greedy_weight + w[vw_ratio[index][0]] < W:
        greedy_val += v[vw_ratio[index][0]]
        greedy_weight += w[vw_ratio[index][0]]
        greedy_bag += [vw_ratio[index][0]]
        index += 1
    
    print("Greedy Algorithm:\nValue:{}, Weight:{}\nBag:{}".format(greedy_val, greedy_weight, greedy_bag))
    SA_trace = simulated_annealing()
    plt.plot([greedy_val]*len(SA_trace), label="Greedy")
    plt.plot(SA_trace, label="SA")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    