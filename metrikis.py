# -*- coding: utf-8 -*-


import numpy


def evklid_lenght(vector_1, vector_2, arguments=0):
    if len(vector_1) != len(vector_2):
        print('Lenght vector A != lenght vector B.') #Проверка мерности каждого вектора
        return -1
    temp_list = []
    for i in range(len(vector_1)):
        temp_list.append((vector_1[i] - vector_2[i]) ** 2)
    return (sum(temp_list))**(0.5)


def mannhetn_lenght(vector_1, vector_2, arguments=0):
    if len(vector_1) != len(vector_2):
        print('Lenght vector A != lenght vector B.')
        return -1
    temp_list = []
    for i in range(len(vector_1)):
        temp_list.append(abs(vector_1[i] - vector_2[i]))#append - кладем в конец списка
    return sum(temp_list)


def rem_metr(vector_1, vector_2, arguments=0):
    if len(vector_1) != len(vector_2):
        print('Lenght vector A != lenght vector B.')
        return -1
    temp_list = []
    for i in range(len(vector_1)):
        temp_list.append(abs(vector_1[i] - vector_2[i]))
    return max(temp_list)


def minkovskiy(vector_1, vector_2, p):
    if len(vector_1) != len(vector_2) or p == 0:
        print('Lenght vector A != lenght vector B or p == zero.')
        return -1
    temp_list = []
    for i in range(len(vector_1)):
        temp_list.append((vector_1[i] - vector_2[i]) ** p)#аналог евклидовой, но с любой степенью
    return (sum(temp_list))**(1 / p)


def camber_metr(vector_1, vector_2, arguments=0):
    if len(vector_1) != len(vector_2):
        print('Lenght vector A != lenght vector B.')
        return -1
    temp_list = []
    for i in range(len(vector_1)):
        temp_list.append(abs(vector_1[i] - vector_2[i]) / (abs(vector_1[i]) + abs(vector_2[i])))
    return sum(temp_list)


#def mohave(vector_1, vector_2, matrix):
#    if len(vector_1) != len(vector_2) or len(matrix) != len(vector_2) or len(matrix[0]) != len(vector_2):
#        print('Lenght vector A != lenght vector B or matrix not sqare.')
#        return -1
#    matrix = numpy.array(matrix)
#    matrix_minus_one = matrix.
    

def to_centr(many_vectors, one_vector, name, arguments=0):
    many_vectors = numpy.array(many_vectors)
    centr = many_vectors.sum(0)#центрируем кучу векторов и делаем сумму равную 0
    
    if name == 'evklid_lenght':
        return evklid_lenght(centr, one_vector)
    elif name == 'mannhetn_lenght':
        return mannhetn_lenght(centr, one_vector)
   elif nam 'rem_metr':
        return rem_metr(centr, one_vector)
    elif name == 'minkovskiy':
        return minkovskiy(centr, one_vector, arguments)
    elif name == 'camber_metr':
        return camber_metr(centr, one_vector)
    
    
def nearly(many_vectors, one_vector, name, arguments=0):
    many_vectors = numpy.array(many_vectors)
    dist = []def nearly(many_vectors, one_vector, name, arguments=0):
    if name == 'evklid_lenght':
        name = evklid_lenght
    elif name == 'mannhetn_lenght':
        name = mannhetn_lenght
    elif name == 'rem_metr':
        name = rem_metr
    elif name == 'minkovskiy':
        name = minkovskiy
    elif name == 'camber_metr':
        name = camber_metr
    for i in many_vectors:
        dist.append(name(one_vector, i, arguments))
    return min(dist)
    

def etalon(etalons_vectors, one_vector, name, arguments=0):
    return nearly(etalons_vectors, one_vector, name, arguments=0)


#def nearly_max(many_vectors, one_vector, arguments=0):   
#    pass


if __name__ == '__main__':
    a = [1,2,3]
    b = [3,2,1]
    p = 4
    matrix = [[1,2,3],[4,5,6],[7,8,9]]
#    b = a
    

    
    print(evklid_lenght(a, b))
    print(mannhetn_lenght(a, b))
    print(rem_metr(a, b))
    print(minkovskiy(a, b, p))
    print(camber_metr(a, b))
    print('\n')

    print(to_centr(matrix, b, 'evklid_lenght'))
    print(to_centr(matrix, b, 'mannhetn_lenght'))
    print(to_centr(matrix, b, 'rem_metr'))
    print(to_centr(matrix, b, 'minkovskiy', p))
    print(to_centr(matrix, b, 'camber_metr'))
    print('\n')

    print(nearly(matrix, b, 'evklid_lenght'))
    print(nearly(matrix, b, 'mannhetn_lenght'))
    print(nearly(matrix, b, 'rem_metr'))
    print(nearly(matrix, b, 'minkovskiy', p))
    print(nearly(matrix, b, 'camber_metr'))
    