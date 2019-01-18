import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

import pandas as pd
import numpy as np  # либа для быстрых вычислений

from matplotlib import pyplot as plt
import seaborn as sns

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QDialog, QApplication, QPushButton, QVBoxLayout, QMessageBox

from PandasModel import *
from Add import *
from Perceptron import *
from mraz_gui_schema import *
from Haming import *
from Kohonen import *
from Nsko import *
from Hebb import *
from alg_kopa import *



class MyWin(QtWidgets.QMainWindow):
    ICONS_PATH = 'icons8-студент-480.ico'
    
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_main_window()
        self.ui.setupUi(self)
        self.ui.load_data_btn.clicked.connect(self.load_data_btn_click)
        self.ui.step_slider.valueChanged.connect(self.value_change_slider)
        self.ui.fit_btn.clicked.connect(self.fit_btn_click)
        self.ui.clear_btn.clicked.connect(self.clear_btn_click)
        self.ui.save_to_file_btn.clicked.connect(self.save_to_file_btn_click)

    def load_data_btn_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Выберите файл для обучения", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.df = pd.read_csv(fileName, engine = 'python')
            model = PandasModel(self.df)
            algorithm_index = self.ui.tabWidget.currentIndex() # Получение индекса текущей страницы QTable
            self.insert_data(algorithm_index, model)
        self.ui.load_data_btn.setChecked(False) 

    def insert_data(self, algorithm_index, model):
        if algorithm_index == 0:
            self.ui.data_tbl.setModel(model)
            count_features = len(self.df.columns[:-1])
            self.ui.step_slider.setMaximum(count_features)
        elif algorithm_index == 1:
            self.ui.data_perceptron_tbl.setModel(model)
        elif algorithm_index == 2:
            self.ui.data_haming_tbl.setModel(model)
        elif algorithm_index == 3:
            self.ui.data_kohonen_tbl.setModel(model)
        elif algorithm_index == 4:
            self.ui.data_nsko_tbl.setModel(model)
        elif algorithm_index == 5:
            self.ui.data_habb_tbl.setModel(model)
        elif algorithm_index == 6:
            self.ui.data_kopa_tbl.setModel(model)


    def value_change_slider(self):
        value = self.ui.step_slider.value()
        self.ui.depth_value_lbl.setText(str(value))
        
    def fit_btn_click(self):
        algorithm_index = self.ui.tabWidget.currentIndex()
        self.fit(algorithm_index)

    def fit(self, algorithm_index):
        if algorithm_index == 0:
            self.fit_add()
        elif algorithm_index == 1:
            self.fit_perceptron()
        elif algorithm_index == 2:
            self.fit_haming()
        elif algorithm_index == 3:
            self.fit_kohonen()
        elif algorithm_index == 4:
            self.fit_nsko()
        elif algorithm_index == 5:
            self.fit_habb()

    def fit_add(self):
        X = self.df.iloc[:, :-1]
        Y = self.df.iloc[:, -1]
        model = LogisticRegression();
        step = self.ui.step_slider.value()
        result_features, costs = Add(X, Y, list(self.df.columns[:-1]), model, int(step), True)
        result_features_str = "\n".join(result_features)
        self.clear_btn_click()
        self.ui.fit_btn.setChecked(False)
        self.ui.best_features_te.append(result_features_str)

    def fit_perceptron(self):
        X = np.array(self.df.iloc[:,:-1])
        y = np.array(self.df.iloc[:,-1])

        uniform_X = uniform_vector(X, y)
        max_epochs = self.ui.max_epochs_sb.value()
        weights, num_epochs = Perceptron(uniform_X, max_epochs)
        self.ui.epochs_num_te.setText(str(num_epochs))
        self.ui.weights_matrix_te.setText(str(weights))

    def fit_haming(self):
        X_etalons = np.array(self.df.iloc[:,:-1]).transpose()
        X_classify = np.array(self.df.iloc[:,:-1])
        label = Haming(X_etalons,X_classify)
        self.ui.label_le.setText(str(label))

    def fit_kohonen(self):
        X = np.array(self.df.iloc[:,:-1])
        y = np.array(self.df.iloc[:,-1])
        weights = Kohonen(X,y)
        self.ui.kohonen_weights_le.setText(str(weights))

    def fit_nsko(self):
        X = np.array(self.df.iloc[:,:-1])
        y = np.array(self.df.iloc[:,-1])
        step = self.ui.nsko_step_dsb.value()
        weights = NSKO_alg(X,y,step)
        self.ui.nsko_weights_te.setText(str(weights))

    def fit_habb(self):
        X = np.array(self.df.iloc[:,:-1])
        y = np.array(self.df.iloc[:,-1])
        bip_y = bipolation(y)
        max_iterations = self.ui.max_epochs_habb_sb.value()
        weights, num_iterations = Hebb(X,bip_y,max_iterations)
        self.ui.habb_count_iter_le.setText(str(num_iterations))
        self.ui.habb_weights_te.setText(str(weights))

    def clear_btn_click(self):
        self.ui.best_features_te.clear()
        self.ui.clear_btn.setChecked(False)
        
    def save_to_file_btn_click(self):
        filename = 'best_features.txt'
        with open(filename, 'w') as file_obj:
            file_obj.write(str(self.ui.best_features_te.toPlainText()))
        self.ui.save_to_file_btn.setChecked(False)
        self.features_saved(filename)
            
    def features_saved(self, filename):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Информация")
        msg.setText("Признаки успешно сохранены в файл " + str(filename))
        msg.setWindowIcon(QtGui.QIcon(self.ICONS_PATH))
        msg.exec_()
            
            
if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_())