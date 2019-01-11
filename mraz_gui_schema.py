# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mraz.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.setEnabled(True)
        main_window.resize(807, 463)
        font = QtGui.QFont()
        font.setPointSize(7)
        main_window.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons8-студент-480.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        main_window.setWindowIcon(icon)
        main_window.setFixedSize(807,473)
        main_window.setToolTipDuration(-9)
        self.tabWidget = QtWidgets.QTabWidget(main_window)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 591, 471))
        self.tabWidget.setObjectName("tabWidget")
        self.add_tab = QtWidgets.QWidget()
        self.add_tab.setObjectName("add_tab")
        self.data_tbl = QtWidgets.QTableView(self.add_tab)
        self.data_tbl.setGeometry(QtCore.QRect(0, 0, 381, 331))
        self.data_tbl.setObjectName("data_tbl")
        self.groupBox = QtWidgets.QGroupBox(self.add_tab)
        self.groupBox.setGeometry(QtCore.QRect(0, 340, 281, 101))
        font = QtGui.QFont()
        font.setFamily("MingLiU_HKSCS-ExtB")
        font.setPointSize(10)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(True)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.groupBox.setFont(font)
        self.groupBox.setAutoFillBackground(False)
        self.groupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.groupBox.setFlat(True)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.step_slider = QtWidgets.QSlider(self.groupBox)
        self.step_slider.setGeometry(QtCore.QRect(0, 60, 241, 22))
        self.step_slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.step_slider.setMinimum(1)
        self.step_slider.setMaximum(11)
        self.step_slider.setSingleStep(1)
        self.step_slider.setPageStep(2)
        self.step_slider.setTracking(True)
        self.step_slider.setOrientation(QtCore.Qt.Horizontal)
        self.step_slider.setInvertedAppearance(False)
        self.step_slider.setInvertedControls(True)
        self.step_slider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.step_slider.setTickInterval(2)
        self.step_slider.setObjectName("step_slider")
        self.depth_text_lbl = QtWidgets.QLabel(self.groupBox)
        self.depth_text_lbl.setGeometry(QtCore.QRect(0, 30, 221, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.depth_text_lbl.setFont(font)
        self.depth_text_lbl.setObjectName("depth_text_lbl")
        self.depth_value_lbl = QtWidgets.QLabel(self.groupBox)
        self.depth_value_lbl.setGeometry(QtCore.QRect(250, 60, 41, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.depth_value_lbl.setFont(font)
        self.depth_value_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_value_lbl.setObjectName("depth_value_lbl")
        self.best_features_te = QtWidgets.QTextEdit(self.add_tab)
        self.best_features_te.setGeometry(QtCore.QRect(390, 20, 181, 311))
        self.best_features_te.setObjectName("best_features_te")
        self.best_features_lbl = QtWidgets.QLabel(self.add_tab)
        self.best_features_lbl.setGeometry(QtCore.QRect(390, 0, 181, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.best_features_lbl.setFont(font)
        self.best_features_lbl.setObjectName("best_features_lbl")
        self.save_to_file_btn = QtWidgets.QPushButton(self.add_tab)
        self.save_to_file_btn.setGeometry(QtCore.QRect(390, 390, 181, 41))
        font = QtGui.QFont()
        font.setFamily("MingLiU_HKSCS-ExtB")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.save_to_file_btn.setFont(font)
        self.save_to_file_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.save_to_file_btn.setCheckable(True)
        self.save_to_file_btn.setObjectName("save_to_file_btn")
        self.clear_btn = QtWidgets.QPushButton(self.add_tab)
        self.clear_btn.setGeometry(QtCore.QRect(390, 340, 181, 41))
        font = QtGui.QFont()
        font.setFamily("MingLiU_HKSCS-ExtB")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.clear_btn.setFont(font)
        self.clear_btn.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.clear_btn.setCheckable(True)
        self.clear_btn.setObjectName("clear_btn")
        self.tabWidget.addTab(self.add_tab, "")
        self.perceptron_tab = QtWidgets.QWidget()
        self.perceptron_tab.setObjectName("perceptron_tab")
        self.tabWidget.addTab(self.perceptron_tab, "")
        self.hamming_tab = QtWidgets.QWidget()
        self.hamming_tab.setObjectName("hamming_tab")
        self.tabWidget.addTab(self.hamming_tab, "")
        self.kohonen_tab = QtWidgets.QWidget()
        self.kohonen_tab.setObjectName("kohonen_tab")
        self.tabWidget.addTab(self.kohonen_tab, "")
        self.nsko_tab = QtWidgets.QWidget()
        self.nsko_tab.setObjectName("nsko_tab")
        self.tabWidget.addTab(self.nsko_tab, "")
        self.habb_tab = QtWidgets.QWidget()
        self.habb_tab.setObjectName("habb_tab")
        self.tabWidget.addTab(self.habb_tab, "")
        self.metrikis_tab = QtWidgets.QWidget()
        self.metrikis_tab.setObjectName("metrikis_tab")
        self.tabWidget.addTab(self.metrikis_tab, "")
        self.load_data_btn = QtWidgets.QPushButton(main_window)
        self.load_data_btn.setGeometry(QtCore.QRect(600, 50, 191, 41))
        font = QtGui.QFont()
        font.setFamily("MingLiU_HKSCS-ExtB")
        font.setPointSize(14)
        self.load_data_btn.setFont(font)
        self.load_data_btn.setCheckable(True)
        self.load_data_btn.setObjectName("load_data_btn")
        self.fit_btn = QtWidgets.QPushButton(main_window)
        self.fit_btn.setGeometry(QtCore.QRect(600, 130, 191, 41))
        font = QtGui.QFont()
        font.setFamily("MingLiU_HKSCS-ExtB")
        font.setPointSize(14)
        self.fit_btn.setFont(font)
        self.fit_btn.setCheckable(True)
        self.fit_btn.setObjectName("fit_btn")
        self.test_btn = QtWidgets.QPushButton(main_window)
        self.test_btn.setGeometry(QtCore.QRect(600, 200, 191, 41))
        font = QtGui.QFont()
        font.setFamily("MingLiU_HKSCS-ExtB")
        font.setPointSize(14)
        self.test_btn.setFont(font)
        self.test_btn.setCheckable(True)
        self.test_btn.setObjectName("test_btn")

        self.retranslateUi(main_window)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Примеры работы алгоритмов"))
        self.groupBox.setTitle(_translate("main_window", "Параметры алгоритма"))
        self.depth_text_lbl.setText(_translate("main_window", "Глубина просматривания вперед:"))
        self.depth_value_lbl.setText(_translate("main_window", "1"))
        self.best_features_te.setHtml(_translate("main_window", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.best_features_lbl.setText(_translate("main_window", "Выделенные признаки"))
        self.save_to_file_btn.setText(_translate("main_window", "Сохранить признаки в файл"))
        self.clear_btn.setText(_translate("main_window", "Очистить содержимое"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.add_tab), _translate("main_window", "Add "))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.perceptron_tab), _translate("main_window", "Персептрон"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.hamming_tab), _translate("main_window", "Алгоритм Хемминга"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.kohonen_tab), _translate("main_window", "Алгоритм Кохонена"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.nsko_tab), _translate("main_window", "НСКО"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.habb_tab), _translate("main_window", "Алгоритм Хебба"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.metrikis_tab), _translate("main_window", "Page"))
        self.load_data_btn.setText(_translate("main_window", "Загрузить данные"))
        self.fit_btn.setText(_translate("main_window", "Обучить модель"))
        self.test_btn.setText(_translate("main_window", "Тестирование"))
