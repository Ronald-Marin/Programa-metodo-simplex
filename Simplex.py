
#!/usr/bin/python3
"""
@author: Ronald Marín
"""

from typing import Dict, List, Tuple
from exceptions import *
from decimal import Decimal as dc
from fractions import Fraction

phase1_iterations = list()
phase2_iterations = list()

def get_min_negative(lst: list):
    negative_values: list = list(filter(lambda x: True if x < 0 else False, lst))
    if len(negative_values):
        return lst.index(min(negative_values))
    else:
        raise NoMinNegativeValue


def get_min_positive(lst: list):
    positive_values: list = list(filter(lambda x: True if x > 0 else False, lst))
    if len(positive_values):
        return lst.index(min(positive_values))
    else:
        raise NoMinPositiveValue


def get_max_positive(lst: list):
    positive_values: list = list(filter(lambda x: True if x > 0 else False, lst))
    if len(positive_values):
        return lst.index(max(positive_values))
    else:
        raise NoMaxPostiveValue


def get_pivot(matrix: List[List[float]], costs: List[float], constraints: List[float], nature: bool):
    """Return index of pivot element as a tuple
        Error handling for this function is handled on the Simplex class: self.phase1() method
        nature: bool
            True for minimisation
            False for maximisation
    """
    m = len(matrix)  # Número de restricciones
    n = len(costs)  # Numero de variables

    if nature:
        # Problema de minimización
        jpivot: int = get_min_negative(costs)  # índice de valor mínimo negativo
        ratios: list = list()
        for i in range(m):
            try:
                ratios.append(constraints[i] / matrix[i][jpivot])
            except ZeroDivisionError:
                ratios.append(-1)  # Agregue un valor negativo arbitrario para que no sea procesado por get_min_positive()
        ipivot: int = get_min_positive(ratios)
        return ipivot, jpivot
    else:
        # Problema de maximización
        jpivot: int = get_max_positive(costs)
        ratios: list = list()
        for i in range(m):
            try:
                ratios.append(constraints[i] / matrix[i][jpivot])
            except ZeroDivisionError:
                ratios.append(-1)

        ipivot = get_min_positive(ratios)
        return ipivot, jpivot


class Simplex:
    def __init__(self, n: int, m: int, a: List[List[float]], b: List[float], constraints: List[str], obj_func: List[
        float], nature: bool = True):
        self.error_message: str = ""
        self.n = n  # número de variables
        self.m = m  # número de restricciones
        self.a = a  # coeficiente de matriz de variables
        self.b = b  # valores de restricciones
        self.constraints = constraints  # tipo de restricciones
        self.obj_func = obj_func  # coeficientes de la función objetiva
        self.nature: bool = nature  # Verdadero para problemas de minimización, Falso para maximización

        self.unknowns: Dict[str, float] = dict()  # variables: x1, x2, ....
        self.shift_vars: Dict[str, float] = dict()  # Variables de cambio
        self.artificial_vars: Dict[str, float] = dict()  # Variables artificiales

        # Variables relacionadas con la construcción de tablas de la fase 1
        self.vars_names: List[str] = list()  # nombres de variables
        self.base_vars_names: List[str] = list()  # nombres de variables base
        self.table: List[List[float]] = list()  # tabla correspondiente a la matriz
        self.table_cost: float = 0  # el costo de la tabla calculado a partir de la función objetivo Za
        self.table_cost_phase2: float = 0   # necesitamos este valor al cambiar a la fase dos
        self.Za: List[List[float], float] = [list(), 0]  # La función objetivo de la fase 1 para minimizar
        self.table_coef: List[float] = list()  # coeficiente de función objetivo en la tabla
        self.table_coef_phase2: List[float] = list()    # este valor también es necesario cuando se cambia a la fase dos
        self.table_constraints: List[float] = list()  # limitaciones de la mesa

        # variables de iteración
        self.phase1_steps = list()
        self.phase2_steps = list()

        """Read each line of given matrix then decide what variables to add"""
        for i in range(self.m):
            if self.constraints[i] == 'lt':
                self.shift_vars['h' + str(i + 1)] = 1
            elif self.constraints[i] == 'gt':
                self.shift_vars['h' + str(i + 1)] = -1
                if self.b[i] != 0:
                    self.artificial_vars['a' + str(i + 1)] = 1
            else:
                if self.b[i] != 0:
                    self.artificial_vars['a' + str(i + 1)] = 1

        # El proceso principal
        self._construct_table_phase1()
        if len(self.artificial_vars):
            result_phase1 = self.phase1()
            if result_phase1:
                print("CONSTRUCTING THE TABLE OF PHASE TWO")
                self._phase1_to_phase2_table()
                self.print_state(True)
                self.phase2()
            else:
                pass
        else:
            print("No Need For Phase 1")
            self._construct_phase2_table()
            self.print_state(False)
            self.phase2()

    def _construct_table_phase1(self):
        """Construct the first table of phase 1"""
        self.table = self.a
        k: int = 0  # shift vars index
        l: int = len(self.shift_vars)  # artificial vars index
        p: int = len(self.shift_vars) + len(self.artificial_vars)

        for i in range(self.m):
            zeros = [0 for _ in range(p)]
            if 'h' + str(i + 1) in self.shift_vars:
                zeros[k] = self.shift_vars['h' + str(i + 1)]
                k += 1
            if 'a' + str(i + 1) in self.artificial_vars:
                zeros[l] = self.artificial_vars['a' + str(i + 1)]
                l += 1
            self.table[i] = self.table[i] + zeros
            self.table_constraints.append(self.b[i])

        # Inicia todas las vars con ceros: incógnitas, cambio, artificial
        for i in range(self.n):
            self.unknowns['x' + str(i + 1)] = 0
        for var in self.shift_vars:
            self.shift_vars[var] = 0
        for var in self.artificial_vars:
            self.artificial_vars[var] = 0
        self.vars_names = list(self.unknowns.keys()) + list(self.shift_vars.keys()) + list(self.artificial_vars.keys())

        # actualizar vars base
        for i in range(self.m):
            if 'a' + str(i + 1) in self.artificial_vars:
                self.artificial_vars['a' + str(i + 1)] = self.b[i]
                self.base_vars_names.append('a' + str(i + 1))
            else:
                if 'h' + str(i + 1) in self.shift_vars:
                    self.shift_vars['h' + str(i + 1)] = self.b[i]
                    self.base_vars_names.append('h' + str(i + 1))

        # Corrija la función objetivo
        p: int = self.n + len(self.shift_vars) + len(self.artificial_vars)  # número de variables
        za: List[float] = [0 for _ in range(p)]  # lista de coeficientes multiplicados por -1 y resumidos
        const_b: float = 0  # valor constante sobre el valor objetivo
        for var in self.artificial_vars:
            i: int = int(var[1]) - 1  # número de línea en la matriz
            tmp: List[float] = [-1 * self.table[i][j] for j in range(p)]
            for j in range(len(tmp)):
                za[j] += tmp[j]
            const_b += self.b[i]

        self.Za[0] = za[:self.n + len(self.shift_vars)]
        self.Za[1] = const_b
        self.table_coef = self.Za[0] + [0 for _ in range(len(self.artificial_vars))]
        self.table_cost = const_b

        self.table_coef_phase2 = self.obj_func[0] + [0 for _ in range(len(self.shift_vars) + len(self.artificial_vars))]
        self.table_cost_phase2 = self.obj_func[1]

        self.store_iterations(
            list(self.vars_names),
            list(self.base_vars_names),
            self.table,
            self.table_constraints,
            self.table_coef,
            self.table_cost,
            True
        )
        self.print_state(True)

    def _construct_phase2_table(self):
        """Construct table of simplex probleme
            This method is called when no artificial variables are needed
            since the operation of setuping the table is redundant the other method : _construct_table_phase2 is called
            on the constructor.
            This mehtod will  update only what is necessary
        """
        p: int = self.n + len(self.shift_vars)
        self.table_coef = self.obj_func[0] + [0 for _ in range(p-self.n)]
        self.table_cost = self.obj_func[1]

    def _phase1_to_phase2_table(self):
        """Construct the table of phase 2 by removing artificial variables and recalculate the objective function"""
        p:int = self.n + len(self.shift_vars)
        tmp: List[float] = [0 for _ in range(p)]
        const_coef: float = self.obj_func[1]
        self.table = [line[:p] for line in self.table]
        self.vars_names = self.vars_names[:p]

        # utilizar el coeficiente transportado de la fase uno
        self.table_coef = self.table_coef_phase2[:self.n + len(self.shift_vars)]
        self.table_cost = const_coef + self._calculate_table_cost(self.unknowns, self.obj_func)

        self.store_iterations(
            list(self.vars_names),
            list(self.base_vars_names),
            self.table,
            self.table_constraints,
            self.table_coef,
            self.table_cost,
            False         
        )
        self.print_state(False)



    def phase1(self):
        """Perform phase 1 iterations"""
        # TODO: completar esta función
        while self.table_cost:
            ipivot: int = 0
            jpivot: int = 0
            pivot: float = 0.0
            p: int = self.n + len(self.shift_vars) + len(self.artificial_vars)

            # Gire y divida su línea por sí mismo
            try:
                ipivot, jpivot = get_pivot(self.table, self.table_coef, self.table_constraints, True)
                pivot = self.table[ipivot][jpivot]
            except NoMinNegativeValue:
                print("Error: END OF PHASE ONE")
                if self.table_cost:
                    print("Cost of table is not Null, No Possible Solution for this problem")
                    self.error_message = "Cost of table is not Null, No Possible Solution for this probleme"
                return False
            except NoMinPositiveValue:
                print("Error: SOMETHING IS WRONG WITH THE CALCULATION, BECAUSE NO MIN OF RATIO IS FOUND")
                self.error_message = "Error: SOMETHING IS WRONG WITH THE CALCULATION, BECAUSE NO MIN OF RATIO IS FOUND"
                break
            except DegeneranceProblem:
                print("Degenerance Problem: Unable to sovle")
                self.error_message = "Degenerance Problem: Unable to sovle"
                break
            except Exception as e:
                raise e

            for i in range(p):
                self.table[ipivot][i] /= pivot
            else:
                self.table_constraints[ipivot] /= pivot

            # Actualice cada línea de la tabla de acuerdo con el pivote.
            for i in range(self.m):
                if i != ipivot:
                    multiplier: float = self.table[i][jpivot]
                    for j in range(p):
                        self.table[i][j] = self.table[i][j] - self.table[ipivot][j] * multiplier
                    else:
                        self.table_constraints[i] -= self.table_constraints[ipivot] * multiplier
            else:
                # actualizar la línea de coeficiente
                multiplier: float = self.table_coef[jpivot]
                for i in range(p):
                    self.table_coef[i] -= self.table[ipivot][i] * multiplier

                # actualizar la otra línea de coeficiente
                multiplier: float = self.table_coef_phase2[jpivot]
                for i in range(p):
                    self.table_coef_phase2[i] -= self.table[ipivot][i] * multiplier

            # Actualizar variables: dejando e ingresando una
            leaving: str = self.base_vars_names[ipivot]
            entering: str = self.vars_names[jpivot]
            self.base_vars_names[ipivot] = entering  # Agregar variable de entrada a las variables base

            # restablecer todas las variables y actualizar de acuerdo con la nueva tabla
            for var in self.unknowns: self.unknowns[var] = 0
            for var in self.shift_vars: self.shift_vars[var] = 0
            for var in self.artificial_vars: self.artificial_vars[var] = 0
            for i in range(self.m):
                var: str = self.base_vars_names[i]
                if var in self.unknowns:
                    self.unknowns[var] = self.table_constraints[i]
                if var in self.shift_vars:
                    self.shift_vars[var] = self.table_constraints[i]
                if var in self.artificial_vars:
                    self.artificial_vars[var] = self.table_constraints[i]

            # Actualizar el costo de la tabla
            self.table_cost = self._calculate_table_cost({**self.unknowns, **self.shift_vars}, self.Za)
            self.table_cost_phase2 = self._calculate_table_cost(self.unknowns, self.obj_func)
        
            self.store_iterations(
                list(self.vars_names),
                list(self.base_vars_names),
                self.table,
                self.table_constraints,
                self.table_coef,
                self.table_cost,
                True
            )
        return True

    def phase2(self):
        """Perform iteratin of phase 2"""

        while True:
            ipivot: int = 0
            jpivot: int = 0
            pivot: float = 0.0
            p: int = self.n + len(self.shift_vars)

            # Gire y divida su línea por sí mismo
            if self.nature:
                # Problema de minimización
                try:
                    ipivot, jpivot = get_pivot(self.table, self.table_coef, self.table_constraints, True)
                except NoMinNegativeValue:
                    print("END OF ALGORITHM (MINIMISATION)")
                    print("SOLUTION: ", end=" ")
                    for var in self.unknowns:
                        print("{}: {}".format(var, self.unknowns[var]), end=" ")
                    print("")
                    break
                except Exception as e:
                    raise e
            else:
                # Problema de maximización
                try:
                    ipivot, jpivot = get_pivot(self.table, self.table_coef, self.table_constraints, False)
                except NoMaxPostiveValue:
                    print('END OF ALGORITHM(MAXIMISATION)')
                    print("SOLUTION: ", self.unknowns)
                    break
                except NoMinPositiveValue:
                    print("Error: SOMETHING IS WRONG WITH THE CALCULATION, BECAUSE NO MIN OF RATIO IS FOUND")
                    self.error_message = "Error: SOMETHING IS WRONG WITH THE CALCULATION, BECAUSE NO MIN OF RATIO IS FOUND"
                    break
                except Exception as e:
                    raise e
            pivot = self.table[ipivot][jpivot]

            # Divida la línea de pivote por pivote
            for i in range(p):
                self.table[ipivot][i] /= pivot
            else:
                self.table_constraints[ipivot] /= pivot

            # Divida otras líneas según el pivote
            for i in range(self.m):
                if i is not ipivot:
                    multiplier: float = self.table[i][jpivot]
                    for j in range(p):
                        self.table[i][j] -= multiplier * self.table[ipivot][j]
                    else:
                        self.table_constraints[i] -= multiplier * self.table_constraints[ipivot]
            else:
                # Actualizar coef de tabla
                multiplier = self.table_coef[jpivot]
                for i in range(p):
                    self.table_coef[i] -= multiplier * self.table[ipivot][i]


            # Actualizar variables: dejando e ingresando una
            leaving: str = self.base_vars_names[ipivot]
            entering: str = self.vars_names[jpivot]
            self.base_vars_names[ipivot] = entering  # Agregar variable de entrada a las variables base

            # restablecer todas las variables y actualizar de acuerdo con la nueva tabla
            for var in self.unknowns: self.unknowns[var] = 0
            for var in self.shift_vars: self.shift_vars[var] = 0
            for i in range(self.m):
                var: str = self.base_vars_names[i]
                if var in self.unknowns:
                    self.unknowns[var] = self.table_constraints[i]
                if var in self.shift_vars:
                    self.shift_vars[var] = self.table_constraints[i]

            # Costo de la tabla de cálculo
            self.table_cost = self._calculate_table_cost(self.unknowns, self.obj_func)
            self.store_iterations(
                list(self.vars_names),
                list(self.base_vars_names),
                self.table,
                self.table_constraints,
                self.table_coef,
                self.table_cost,
                False
            )
            self.print_state(False)


    ########################################################################
    # Métodos de propósito general
    ########################################################################

    def _calculate_table_cost(self, vars_names: Dict[str, float], Za: List[float]):
        """Calculate the table cost on phase 1"""
        res: float = Za[1]
        coef: list = Za[0]
        for key,value in list(zip(list(vars_names.keys()), coef)):
            res += vars_names[key] * value

        return res

    def store_iterations(
        self,
        vars_names,
        base_vars_names,
        table,
        table_constraints,
        table_coef,
        table_cost,
        phase
    ):

        # resolver el problema de las variables anuladas
        variables_names = [var for var in vars_names]
        base_variables_names = [var for var in base_vars_names]
        matrix_table = [[var for var in line] for line in table]
        constraints = [var for var in table_constraints]
        coefs = [var for var in table_coef]
        cost = table_cost

        if phase:


            self.phase1_steps.append([
                variables_names,
                base_variables_names,
                matrix_table,
                constraints,
                coefs,
                cost
            ])
        else:
            self.phase2_steps.append([
                variables_names,
                base_vars_names,
                matrix_table,
                constraints,
                coefs,
                cost
            ])

    def print_state(self, nature: bool):
        # print("Unknows: ", self.unknowns)
        print("Unknowns: ", end="{")
        for var in self.unknowns:
            print("{}: {} ".format(var, self.unknowns[var]), end="")
        print("}")

        # print("shift Variables:", self.shift_vars)
        print("Shift Vars: ", end="{")
        for var in self.shift_vars:
            print("{}: {} ".format(var, self.shift_vars[var]), end="")
        print("}")

        if nature:
            # print("Artificial vars:", self.artificial_vars)
            print("Artificial Variable: ", end="{")
            for var in self.artificial_vars:
                print("{}: {}, ".format(var, self.artificial_vars[var]), end="")
            print("}")

        # print("*", self.vars_names, "constraints")
        print("*", end=" | ")
        for var in self.vars_names:
            print("{}".format(var), end="\t")
        else:
            print(" | Bi")

        for i in range(self.m):
            # print(self.base_vars_names[i], self.table[i], self.table_constraints[i])
            print(self.base_vars_names[i], end=" | ")
            for var in self.table[i]:
                print("{}".format(var), end="\t")
            else:
                print(" | {}".format(self.table_constraints[i]))

        # print("costs", self.table_coef, self.table_cost)
        print("Z ", end=" | ")
        for var in self.table_coef:
            print("{}".format(var), end="\t")
        else:
            print(" | {}".format(self.table_cost))

        print("=" * 20)


if __name__ == '__main__':
    # problem with aritificl variables
    # n: int = 2  # nombre de variables
    # m: int = 4  # nombre de contrainte
    # a = [
    #     [Fraction(10), Fraction(5)],
    #     [Fraction(2), Fraction(3)],
    #     [Fraction(1), Fraction(0)],
    #     [Fraction(0), Fraction(1)]
    # ]
    # b = [Fraction(200), Fraction(60), Fraction(12), Fraction(6)]
    # const = ['lt', 'eq', 'lt', 'gt']
    # objective_function = [Fraction(1000), Fraction(2000)]
    # simplex = Simplex(n, m, a, b, const, [objective_function, 0], False)
    #
    # print("+"*100)
    # print("ANOTHER EXAMPLE")
    # print("+"*100)
    # probleme with aritficial variables
    # n: int = 2
    # m: int = 4
    # a = [
    #     [2, 1],
    #     [1, 1],
    #     [5, 4],
    #     [1, 2]
    # ]
    # b = [600, 225, 1000, 150]
    # const = ['lt', 'lt', 'lt', 'lt']
    # objective_function = [3, 4]

    # probelem without artificial variables
    # n: int = 2
    # m: int = 3
    # a = [
    #     [1, 0],
    #     [0, 2],
    #     [3, 2]
    # ]
    # b = [4, 12, 18]
    # const = ['lt', 'lt', 'lt']
    # objective_function = [3, 5]
    n:int = 2
    m: int = 4
    a = [
        [Fraction(1, 10), Fraction(0)],
        [Fraction(0), Fraction(1, 10)],
        [Fraction(1, 10), Fraction(2, 10)],
        [Fraction(2, 10), Fraction(1, 10)]
    ]
    b = [Fraction(4, 10), Fraction(6, 10), Fraction(2), Fraction(17, 10)]
    const = ['gt', 'gt', 'gt', 'gt']
    objective_function = [Fraction(100), Fraction(40)]
    simplex = Simplex(n, m, a, b, const, [objective_function, 0], True)
    print(len(simplex.phase1_steps))
    for iteration in simplex.phase1_steps:
        print(iteration)
        print("="*100)

    # print("pivot Minimisation: ", get_pivot([
    #     [10, 5, 1, 0, 0, 0, 0, 0],
    #     [2, 3, 0, 1, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 0, -1, -1]
    # ], [-2, -4, 0, 0, 0, 1, 0], [200, 60,12, 6], True))
    #
    # print("pivot maximisation: ", get_pivot([
    #     [1, 2, 3, 1, 0, 0],
    #     [15, 21, 30, 0, 1, 0],
    #     [1, 1, 1, 0, 0, 1]
    # ], [87, 147, 258, 0, 0, 0], [90, 1260, 84, 0], False))
    print("End of execution")
