import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

def get_dis(a, b):
    return np.linalg.norm(a - b)

def equal(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if (a[i] != b[i]).any():
            return False
    return True

def kmeans(lt, k_Num, max_Iter=200): # remember to change the maxIter times as you want !!!!
    if k_Num > len(lt):
        print('the number of the set is less than the number you want to cluster which named as k.')
        return lt

    # get a rand set whose length is k_Num from lt
    lst = list(range(len(lt)))
    random.seed()
    random.shuffle(lst)
    C = []
    for i in range(k_Num):
        C.append(lt[lst[i]])
    # start the iteration
    Max_Iter = max_Iter
    while max_Iter > 0:
        max_Iter -= 1
        old_C = copy.copy(C)
        Sets = [{} for i in range(k_Num)]
        belongs = [0 for i in range(len(lt))]
        print('Iter = %d' %(Max_Iter - max_Iter))
        # get each one's belonging
        for i in range(len(lt)):
            belong = 0
            now = lt[i]
            distance = get_dis(C[0], now)
            for j in range(1, k_Num):
                cnt_distance = get_dis(C[j], now)
                if cnt_distance < distance:
                    belong = j
                    distance = cnt_distance
            Sets[belong][i] = 1
            belongs[i] = belong
            if i % 10000 == 0:
                print('    max_Iter = %d , (%d %% %d) have done.' % (Max_Iter - max_Iter, i, len(lt)))
        print('  we have got each one\'s belongs')

        # get the new core of each set
        for i in range(k_Num):
            core = C[i] - C[i]
            for key in Sets[i].keys():
                core = core + lt[key]
            if len(Sets[i]) > 0:
                core = core / len(Sets[i])
            C[i] = core
        print('  we have got new core of each set')

        # if the set C does't maintain, break
        if equal(old_C, C):
            break
    return C, belongs

def get_rand(dem=2):
    return np.random.randint(low = -1000, high = 1000, size = [dem])

if __name__ == '__main__':
    print('try it.')
    '''
    lt = []
    k_num = 5
    number = 10000
    dem = 2
    core = [get_rand(dem) for i in range(k_num)]
    for i in range(number):
        belong = i // (number // k_num)
        point = core[belong] + np.random.randint(low = 0, high = 200, size = [dem])
        lt.append(point)
    '''
    lt = [np.array([1, 1, 1]), np.array([1, 1, 2]), np.array([1, 1, 4]), np.array([100, 100, 100])]
    k_num = 2
    Sets, belongs = kmeans(lt, k_num)
    print(Sets)
    print(belongs)
    plt.plot(belongs)
#    plt.show()

