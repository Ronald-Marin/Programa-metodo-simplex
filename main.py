
#!/usr/bin/python3
"""
@author: Ronald Marín
"""

from PyQt5 import QtCore, QtWidgets, QtGui
from typing import List, Dict, Tuple
from Simplex import Simplex
from fractions import Fraction

def float2fraction(number):
    if type(number) == float:
        decimal: float = number % 1
        multiplier: int = 10 ** (len(str(decimal)) - 2)
        return Fraction(int(number * multiplier), multiplier)

    if type(number) == int:
        return Fraction(number, 1)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.window_height, self.window_width = 400, 600
        font = QtGui.QFont()
        font.setPointSize(20)
        self.number_of_variables: int = 0
        self.number_of_constraints: int = 0
        self.child_window: Inputwidget = None

        # estilo de ventana
        # self.resize(self.window_width, self.window_height)
        self.setWindowTitle("Metodo Simplex")
               
        # estilo de encabezado
        self.title = QtWidgets.QLabel("Metodo Simplex", self)
        self.title.setFont(font)

        # numero de estilo de vaiables
        self.var_number_lable = QtWidgets.QLabel("Numero de Variables:", self)
        self.var_number_input = QtWidgets.QSpinBox(self)

        # numero de estilo de restricciones 
        self.var_constraint_label = QtWidgets.QLabel("Numero de restricciones: ", self)
        self.var_constraint_input = QtWidgets.QSpinBox(self)

        # botones
        self.generate_button = QtWidgets.QPushButton("Generar", self)
        self.exit_button = QtWidgets.QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.exit_app)
        self.generate_button.clicked.connect(self.get_user_input)

        # gestión de diseño
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.title)

        formgrid = QtWidgets.QFormLayout()
        formgrid.addRow(self.var_number_lable, self.var_number_input)
        formgrid.addRow(self.var_constraint_label, self.var_constraint_input)
        vbox.addLayout(formgrid)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.exit_button)
        hbox.addWidget(self.generate_button)

        vbox.addLayout(hbox)

        self.setLayout(vbox)

    # ranura: salir de la aplicación
    def exit_app(self):
        self.close()
        exit()

    # ranura: obtener entrada
    def get_user_input(self):
        self.number_of_variables = int(self.var_number_input.text())
        self.number_of_constraints = int(self.var_constraint_input.text())

        self.child_window: Inputwidget = Inputwidget(self.number_of_variables, self.number_of_constraints)
        self.child_window.show()

class Inputwidget(QtWidgets.QWidget):
    def __init__(self, n: int, m: int): 
        super().__init__()
        self.n: int = n  # Número de variables
        self.m: int = m  # Número de restricciones
        self.nature: bool = False  # Falso para minimización y verdadero para maximización
        self.a: List[List[float]] = list()  # matriz de restricciones y coeficientes
        self.boundry: List[float] = list()  # variables del segundo miembro
        self.constraints: List[str] = list()  # restricciones de la matriz
        self.objective_function = list()
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.solution_window: SolvingWidget = None
        self.setWindowTitle("Metodo Simplex")
        
        # Disposición de la función objetiva
        func_layout = QtWidgets.QHBoxLayout()
        func_layout.addWidget(QtWidgets.QLabel("Funcion Objetivo: ", self))
        var_names: List[str] = ['x'+str(i) for i in range(1, n+1)]
        self.var_inputs: List[QtWidgets.QLineEdit] = list()
        for _ in range(self.n):
            input_box = QtWidgets.QLineEdit("0", self)
            input_box.setValidator(QtGui.QDoubleValidator())
            self.var_inputs.append(input_box)

        for name, input_box in zip(var_names, self.var_inputs):
            func_layout.addWidget(input_box)
            func_layout.addWidget(QtWidgets.QLabel(name, self), 1)

        # diseño de cuadrícula para matriz
        matrix_layout = QtWidgets.QGridLayout()
        self.input_matrix: List[List[QtWidgets.QLineEdit]] = list()
        self.b: List[QtWidgets.QLineEdit] = list()
        self.constraints_inputs: List[QtWidgets.QComboBox] = list()
        for i in range(self.m):
            input_line: List[QtWidgets.QLineEdit] = list()
            for j in range(self.n):
                var_input = QtWidgets.QLineEdit("0", self)
                var_input.setValidator(QtGui.QDoubleValidator())
                input_line.append(var_input)
            else:
                self.input_matrix.append(input_line)

            # agregar entrada de límites
            tmp_input: QtWidgets.QLineEdit = QtWidgets.QLineEdit("0", self)
            tmp_input.setValidator(QtGui.QDoubleValidator())
            self.b.append(tmp_input)

            # agregar entrada de restricción
            tmp_combo = QtWidgets.QComboBox(self)
            tmp_combo.addItems(['<', '>', '='])
            self.constraints_inputs.append(tmp_combo)


        for i in range(self.m):
            for j in range(0, self.n):
                matrix_layout.addWidget(self.input_matrix[i][j], i, j)
            else:
                matrix_layout.addWidget(self.constraints_inputs[i], i, self.n + 1)
                matrix_layout.addWidget(self.b[i], i, self.n + 2)

        # tipo de problema
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel("Objetivo: ", self))
        self.problem_nature: QtWidgets.QComboBox = QtWidgets.QComboBox(self)
        self.problem_nature.addItems(['Minimizacion', 'Maximizacion'])
        hbox.addWidget(self.problem_nature)
        hbox.addStretch()

        # botones: salir y resolver
        button_layout = QtWidgets.QHBoxLayout()
        exit_button = QtWidgets.QPushButton("Exit", self)
        solve_button = QtWidgets.QPushButton("Resolver", self)
        button_layout.addStretch()
        button_layout.addWidget(exit_button)
        button_layout.addWidget(solve_button)
        exit_button.clicked.connect(self.exit_app)
        solve_button.clicked.connect(self.solve_problem)

        self.main_layout.addLayout(func_layout)
        self.main_layout.stretch(1)
        self.main_layout.addLayout(hbox)
        self.main_layout.addWidget(QtWidgets.QLabel("Restricciones"))
        self.main_layout.addLayout(matrix_layout)
        self.main_layout.addLayout(button_layout)
        self.setLayout(self.main_layout)

    # Ranura: botón de salida
    def exit_app(self):
        self.close()

    # Ranura: resuelve el problema
    def solve_problem(self):
        """Obtener entrada del usuario"""
        # restablecer todas las variables
        self.a = list()
        self.objective_function = list()
        self.boundry = list()
        self.constraints = list()

        # obtener entrada de matriz
        for i in range(self.m):
            tmp_line: List[float] = []
            for j in range(self.n):
                inp: float = float(self.input_matrix[i][j].text())
                tmp_line.append(float2fraction(inp))
            else:
                # almacenar línea y obtener límites y valores de límites
                self.a.append(tmp_line)
                self.boundry.append(float2fraction(float(self.b[i].text())))
                constraint: str = str(self.constraints_inputs[i].currentText())
                if constraint == '>':
                    self.constraints.append('gt')
                elif constraint == '<':
                    self.constraints.append('lt')
                else:
                    self.constraints.append('eq')

        # obtener el valor de la función objetivo y la naturaleza del problema
        for i in range(self.n):
            self.objective_function.append(float2fraction(float(self.var_inputs[i].text())))
        nature = int(self.problem_nature.currentIndex())
        self.nature = True if nature else False

        if self.verification():
            self.solution_window = SolvingWidget(
                self.n,
                self.m,
                self.a,
                self.boundry,
                self.constraints,
                self.objective_function,
                self.nature
            )
            self.solution_window.show()


    def verification(self):
        """Verificar la entrada del usuario"""
        alert = QtWidgets.QMessageBox()
        alert.setIcon(QtWidgets.QMessageBox.Critical)
        alert.setWindowTitle("Error")


        if not any(self.objective_function):
            alert.setText("Funcion objetivo")
            alert.setInformativeText("La función objetivo está vacía!!!")
            alert.exec_()
            return False

        if any(list(filter(lambda x: True if x < 0 else False, self.boundry))):
            alert.setText("Error de límites")
            alert.setInformativeText("All boundry should be positive.\nYou may wanna multiply be -1")
            alert.exec_()
            return False
        return True


class SolvingWidget(QtWidgets.QScrollArea):
    def __init__(self, n, m, a, b, constraints, objective_function, nature=False):
        super(SolvingWidget, self).__init__()
        self.solution = Simplex(n, m, a, b, constraints, [objective_function, 0], not nature)
        self.setWindowTitle("Solución simplex")

        self.widget = QtWidgets.QWidget()  # main widget that contains content
        main_layout = QtWidgets.QVBoxLayout(self.widget)
        main_layout.addWidget(QtWidgets.QLabel("Solucion"))

        phase1: List = self.solution.phase1_steps if self.solution.phase1_steps else False
        iteration: int = 1
        if phase1:
            for iteration_table in phase1:
                main_layout.addWidget(QtWidgets.QLabel(str(iteration)))
                iteration += 1
                main_layout.addWidget(self._construct_table(iteration_table))
        else:
            main_layout.addWidget(QtWidgets.QLabel("La fase uno no es necesaria !!!"))


        main_layout.addWidget(QtWidgets.QLabel("Fase dos"))
        iteration = 1
        phase2 = self.solution.phase2_steps if self.solution.phase2_steps else False
        if phase2:
            for iteration_table in phase2:
                main_layout.addWidget(QtWidgets.QLabel(str(iteration)))
                iteration += 1
                main_layout.addWidget(self._construct_table(iteration_table))
        else:
            main_layout.addWidget(QtWidgets.QLabel("No necesita fase, ya está en la solución óptima"))

        if len(self.solution.error_message):
            main_layout.addWidget(QtWidgets.QLabel(self.solution.error_message))
            print(self.solution.error_message)


        self.setWidget(self.widget)
        self.setWidgetResizable(True)
        self.showMaximized()

    def _construct_table(self, iteration_table):
        """Crear un widget de tabla para la solución"""
        n: int = len(iteration_table[0])
        m: int = len(iteration_table[1])

        table_widget = QtWidgets.QTableWidget(m + 1, n + 1, self.widget)
        table_widget.setHorizontalHeaderLabels(list( iteration_table[0] + ['Bi']))
        table_widget.setVerticalHeaderLabels(iteration_table[1]+ ['Z'])

        for i in range(m):
            for j in range(n):
                table_widget.setItem(i, j, QtWidgets.QTableWidgetItem(str(iteration_table[2][i][j])))
            else:
                # Agregar restricciones
                table_widget.setItem(i, n, QtWidgets.QTableWidgetItem(str(iteration_table[3][i])))
        else:
            # Agregar coeficiente de función objetivo
            for i in range(n):
                table_widget.setItem(m, i, QtWidgets.QTableWidgetItem(str(iteration_table[4][i])))
            else:
                table_widget.setItem(m, n, QtWidgets.QTableWidgetItem(str(iteration_table[5])))

        table_widget.setMinimumHeight(m * 50)
        return table_widget
    
    def close_window(self):
        self.solution = None
        self.close()

def main():
    app = QtWidgets.QApplication([])
    mainwindow = MainWindow()
    mainwindow.show()
    app.exec_()


if __name__ == '__main__':
    main()
