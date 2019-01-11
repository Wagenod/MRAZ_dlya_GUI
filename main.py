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
from mraz_gui_schema import *



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
            self.ui.data_tbl.setModel(model) #добавляем в таблицу данные
            count_features = len(self.df.columns[:-1])
            self.ui.step_slider.setMaximum(count_features)
        self.ui.load_data_btn.setChecked(False) 
            
    def value_change_slider(self):
        value = self.ui.step_slider.value()
        self.ui.depth_value_lbl.setText(str(value))
        
    def fit_btn_click(self):
        X = self.df.iloc[:,:-1]
        Y = self.df.iloc[:,-1]
        model = LogisticRegression();
        step = self.ui.step_slider.value()
        result_features, costs = Add(X, Y, list(self.df.columns[:-1]),model, int(step), True)
        
        result_features_str = "\n".join(result_features)
        self.clear_btn_click()
        self.ui.fit_btn.setChecked(False)
        self.ui.best_features_te.append(result_features_str)
        
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