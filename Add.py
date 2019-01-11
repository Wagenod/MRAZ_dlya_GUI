
# coding: utf-8

# In[5]:


#импортируем необходимые функции
from sklearn.linear_model import LogisticRegression #  LogisticRegression - класс для построения логистической регрессии
from sklearn.model_selection import train_test_split # train_test_split - функция, которая разбивает всю выборку на
                                                    # тестовую(контрольную) и обучающую в заданном соотношении
import pandas as pd #c помощью библиотеки Pandas табличные данные очень удобно загружать, обрабатывать 
                    #и анализировать с помощью SQL-подобных запросов.
import numpy as np  # либа для быстрых вычислений
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns



def cross_entropy_cost(y_pred, y_true):
    """
        Данная функция реализует вычисление значения внешнего критерия для тестовой выборки X_k

        Arguments:
        y_pred - предсказание модели (В случае логистической - вероятности принадлежности объектов к 1 классу)
        y_true - для метки классов

    """
    k = y_pred.shape[0] #количество объектов
    # это может быть трудновато для понимания. Здесь используется  техника broadcasting, позволяющая складывать матрицы разных
    # размеров, к вектору прибавлять число и тп.
    # Например, запись 1 - [2,4,1,5,4] равносильна [1-2, 1-4, 1-1, 1-5, 1-4]
    return np.sum(- y_true*np.log(y_pred) - (1 - y_true)*np.log(1-y_pred))/k 

def get_best_feature(F, G, model, X_L_splitted):
    """
        Данная функция реализует поиск признака наиболее выгодного для добавления к списку G 

        Arguments:

        Type          Name         Description 
        list          G -          список уже ранее отобранных "хороших" признаков
        list          X_L_splitted список из следующих элементов (по порядку):
                                                    X_l - обучающую выборка
                                                    X_k - тестовая выборка
                                                    y_l - метки классов для X_l
                                                    y_k - метки классов для X_k
        list          F -          список признаков, из которых нужно выбрать оптимальный
                      model -      модель(классификатор), для которой(-го) осуществляется поиск информативных признаков
        
        Returns - список f_best из двух элементов: f_best[0] - оптимальный для добавления признак
                                                   f_best[1] - нижняя огибающая критерия для модели текущей сложности
    """  
    X_l, X_k, y_l, y_k = X_L_splitted # равносильно следующему: X_l = X_L_splitted[0], X_k = X_L_splitted[1] и тд., но намного короче
    f_best = [-1,20000] ; # оптимальный признак и его значение внешнего критерия на текущем шаге алгоритма (временный список)
    for feature in F:
        G_plus_feature = list(G)
        G_plus_feature.append(feature) # аналогично операции G_(j−1) ∪ f с шага 3
        
        model.fit(X_l[G_plus_feature],y_l) # обучение модели
        y_predict = model.predict_proba(X_k[G_plus_feature]) #классификация объектов тестовой выборки с помощью обученой модели
        cost = cross_entropy_cost(y_predict[:,1],y_k) # вычисление Q (G_(j−1) ∪ f)
        if cost < f_best[1]:
            f_best[0] = feature;
            f_best[1] = cost;
    return f_best; # f*


def plot_curve(costs):
    """
        Данная функция осуществляет построение огибающей кривой
        
        Arguments:
        Type    Name     Description
        list    costs -  список значений внешнего критерия
    """
    sns.set(); # устанавливаем стиль seaborn-а, чтобы красиво было
    plt.rcParams['figure.figsize'] = 15,8
    plt.plot(range(1,len(costs) + 1),costs, marker = '.', ms = 15, ls = '--');
    plt.title("Нижняя огибающая критерия", fontsize = 18)
    plt.xlabel("Сложность модели,j ", fontsize = 18)
    plt.ylabel("$ Q(j) = \min_{\mathscr{F} : |\mathscr{F}| = j} Q (\mathscr{F}) $", fontsize = 16)
    plt.xticks(size = 12)
    plt.yticks(size = 12)
    plt.xlim(1,len(costs))
    plt.ylim(np.min(costs) - 0.01, np.max(costs) + 0.01)
    idx_min = np.argmin(costs)
    plt.axvline(x = idx_min + 1, ymax = 0.01/(np.max(costs) - np.min(costs) + 0.02), color='red', )
    plt.axhline(y = costs[idx_min], xmax = (idx_min + 1 - 1)/(len(costs) - 1), color='red')
    plt.show();
    
    
    
def Add(X_L, Y, F, model, d =2, is_plot_curve = False):
    """
        Данная функция реализует алгоритм Add поиска информативных признаков

        Arguments:

        Type          Name        Description 
        DataFrame     X_L -       выборка (целиком = test + train)
        np.array      Y -         метки классов для каждого объекта выборки (Y = {0,1}) 
        list          F -         список признаков, которыми описываются объекты
        int           d -         параметр, определяющий глубину просматривания вперед (по умолчанию равен 2)
                      model -     модель(классификатор), для которой(-го) осуществляется поиск информативных признаков
        bool          plot_curve- флаг принимает True, если необходимо построить нижнюю огибающую критерия (по умолчанию равен False)
        Returns - список оптимальных признаков
    """   
    X_L_splitted = train_test_split(X_L, Y, test_size = 0.3, random_state = 42) # разбиение выборки на тестовую и обучающую
    j_best = -1;   # оптимальная сложность модели
    G = list();    # сюда будем записывать признаки оптимальные на каждом шаге алгоритма (вначале он пустой G_0 = 0)
    costs = list() # сюда будем записывать значения внешнего критерия
    n = len(F)     # количество признаков
    for j in range(1,n+1): # Реализация "для всех  j=1,2,...,n,где j  — сложность наборов"
        
        f_best = get_best_feature(F, G, model, X_L_splitted) # f* = arg min Q( G_(j−1) ∪ f)   
        #добавление признака в список оптимальных
        G.append(f_best[0]); # G_j = G_(j-1) ∪ f*
        costs.append(f_best[1])
        del F[F.index(f_best[0])]; # удаление признака f_best из списка F. Код F.index(f_best[0]) возвращает индекс элемента
        # f_best[0] в списке F
        
        # Сначала приводим список (list) к нампаевскому массиву, чтобы потом воспользоваться встроенной функцией поиска 
        # минимального элемента, но поскольку индексация в массиве начинается с 0, а сложность модели с 1, то найденный 
        # индекс на 1 меньше сложности модели. Поэтому оптим. сложность модели = "оптим. индекс" + 1 
        j_best = np.argmin(np.array(costs)) + 1 # j* = arg min Q(G_s) для s<=j  
        
        if (j - j_best) > d:
            if is_plot_curve:
                plot_curve(costs)
            return G[:j_best], costs;
    if is_plot_curve:
        plot_curve(costs)

    return G, costs;

#if __name__ =="__main__":
#    df = pd.read_csv('C:\\Users\\PawlikPython\\telecom_churn.csv') 
#    
#    df.drop('State', inplace=True, axis = 1) # удаляем столбец (признак) State, поскольку в нем достаточно много уникальных 
                                    # значений и с этим непонятно, что делать, а выполнение one-hot encoding даст очень много
                                    # новых признаков
#    mapping = {"Yes": 1, "No":0} 
#    df['International plan'] = df['International plan'].map(mapping) # заменяем "Yes" на 1, а "No" на 0
#    df['Voice mail plan'] = df['Voice mail plan'].map(mapping) # --//--
#    df['Churn'] = df['Churn'].astype(dtype='int') # True -> 1, False -> 0
#    
#    X = df.iloc[:,:-1]
#    Y = df.iloc[:,-1]
#    model = LogisticRegression();
#    result_features, costs = Add(X, Y, list(df.columns[:-1]),model, 8, True)
#    print(result_features)

