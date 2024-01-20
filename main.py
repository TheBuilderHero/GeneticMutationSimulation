import numpy as np
import numpy.random as npr
import random
import re

pop_size = 30  # initial population size
max_gen = 50   # cap for number of generations
p_cross = 0.9  # chance of a crossover occurring
p_mut = 0.05   # chance of a mutation


# https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
def roulette_wheel(pop, fit_scores):  # proportional selection function
    invert_fitness = [1.0 / f for f in fit_scores]  # have to invert for minimization
    fit_sum = sum(invert_fitness)  # will be close to "1.0"
    selection_prob = [f / fit_sum for f in invert_fitness]
    return population[npr.choice(len(pop), p=selection_prob)]  # higher number twice as likely to be selected


def sin_point_mut(population, mutation_rate=0.05):
    mutation_amount = random.uniform(-mutation_rate, mutation_rate)  # specifies the range of the mutation
    mutated_value = population + mutation_amount                     # adds the random amount to the genome
    return mutated_value


def remove_dots(input_str):
    return re.sub(r'\.{2,}', '.', input_str)  # regex to remove dots in case there's two

# https://www.geeksforgeeks.org/python-single-point-crossover-in-genetic-algorithm/
def crossover(a, b):
    # converts floats to char list
    str_a = str(a)
    a_list = list(str_a)

    str_b = str(b)
    b_list = list(str_b)

    # stops problem of strings being two different sizes by adding zeros to the list
    max_length = max(len(a_list), len(b_list))
    a_list += ['0'] * (max_length - len(a_list))
    b_list += ['0'] * (max_length - len(b_list))

    k = random.randint(0, len(a_list))  # creates a cross-over point
    # print("Crossover point :", k)

    # interchanging the genes
    for i in range(k, len(a_list)):
        a_list[i], b_list[i] = b_list[i], a_list[i]

    a_list = ''.join(a_list)
    b_list = ''.join(b_list)

    a_list = remove_dots(a_list)
    b_list = remove_dots(b_list)

    return float(a_list), float(b_list)  # returns new float numbers after breeding


def f_equation(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2  # function for assigning fitness using minimum


if __name__ == "__main__":
    population = []  # empty array to store fitness values from generated bounds
    fit_values = []
    mut_values = []

    new_gen = []

    genes = np.random.uniform(low=-5.0, high=4.0, size=(pop_size, 3))  # 30 rows of 3 generated

    for gen in range(max_gen):
        population = np.apply_along_axis(f_equation, 1, genes)  # 'axis=1' for performing an operation per row

        pop_sum = np.sum(population)
        fit_values = population / pop_sum  # sum of new array will be close to "1.0"

    for i in range(15):
        while True:
            par_a = roulette_wheel(population, fit_values)
            par_b = roulette_wheel(population, fit_values)

            if par_a != par_b:  # can't mate with itself
                break

        new_gen = np.append(new_gen, crossover(par_a, par_b))
        mut_values = [sin_point_mut(value, p_mut) for value in new_gen]

    print("First gen:")
    print(population)
    print("\nNew gen:")
    print(np.array(mut_values))

    # print(np.sum(fit_values))
