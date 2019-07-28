import random
import math
import numpy as np
import matplotlib.pyplot as plt

def get_dis(a, b):
    # exli distance
    ret = 0.0
    for i in range(len(a)):
        ret += (a[i] - b[i]) ** 2
    ret = math.sqrt(ret)
    return ret

def kmeans(lt, k_Num, max_Iter=500): # remember to change the maxIter times as you want !!!!
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

    print('Iter start:')
    havedIter = 0
    # start the iteration
    while max_Iter > 0:
        havedIter += 1
        old_C = C
        Sets = [{} for i in range(k_Num)]
        belongs = [0 for i in range(len(lt))]

        print('  Iter %d, determain the belongs ' % (havedIter))
        # get each one's belonging
        for i in range(len(lt)):
            belong = 0
            distance = get_dis(C[0], lt[i])
            for j in range(1, k_Num):
                cnt_distance = get_dis(C[j], lt[i])
                if cnt_distance < distance:
                    belong = j
                    distance = cnt_distance
            Sets[belong][i] = 1
            belongs[i] = belong
            if i % 10000 == 0:
                print('    Iter %d: %d have determained' % (havedIter, i))

        print('  Iter %d, maintain the set' % (havedIter))
        # get the new core of each set
        for i in range(k_Num):
            core = C[i] - C[i]
            for key in Sets[i].keys():
                core = core + lt[key]
            if len(Sets[i]) > 0:
                core = core / len(Sets[i])
            C[i] = core

        # show the process to user
        print('%d Iters have been done, the rest is no more than %d ' % (havedIter, max_Iter))

        # if the set C does't maintain, break
        if old_C == C:
            break
    return C, belongs

def get_rand(dem=2):
    return np.random.randint(low = -1000, high = 1000, size = [dem])

if __name__ == '__main__':
    print('try it.')
    lt = []
    k_num = 5
    number = 10000
    dem = 2
    core = [get_rand(dem) for i in range(k_num)]
    for i in range(number):
        belong = i // (number // k_num)
        point = core[belong] + np.random.randint(low = 0, high = 200, size = [dem])
        lt.append(point)
    Sets, belongs = kmeans(lt, k_num)
    print(core)
    print(Sets)
    print(belongs)
    plt.plot(belongs)
    plt.show()

