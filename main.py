import numpy as np
import numpy.random as npr
import random
import re

pop_size = 30  # initial population size (N)
max_gen = 50   # cap for number of generations
p_mut = 0.05   # chance of a mutation


# https://stackoverflow.com/questions/10324015/fitness-proportionate-selection-roulette-wheel-selection-in-python
def roulette_wheel(pop, fit_scores):                # proportional selection function
    invert_fitness = [1.0 / f for f in fit_scores]  # have to invert for minimization
    fit_sum = sum(invert_fitness)                   # will be close to "1.0"
    selection_prob = [f / fit_sum for f in invert_fitness]
    return population[npr.choice(len(pop), p=selection_prob)]  # higher number twice as likely to be selected


def mutation(pop):
    mutation_amount = random.uniform(pop - 1, pop + 1)  # specifies the range of the mutation
    mutated_value = pop + mutation_amount               # adds the random amount to the genome
    return mutated_value


def remove_dots(a_str, b_str):
    match_1 = re.search(r'\.{2}', a_str)  # checks both strings for ".."
    match_2 = re.search(r'\.{2}', b_str)  # removes a "." and place in other string if found

    if match_1:
        index = match_1.start()
        a_str = re.sub(r'\.{2,}', '.', a_str)

        b_str = b_str[:index] + '.' + b_str[index:]
        return a_str, b_str
    elif match_2:
        index = match_2.start()
        b_str = re.sub(r'\.{2,}', '.', b_str)

        a_str = a_str[:index] + '.' + a_str[index:]
        return a_str, b_str
    else:
        return a_str, b_str


# https://www.geeksforgeeks.org/python-single-point-crossover-in-genetic-algorithm/
def crossover(a, b):
    if random.random() < 0.1:  # 10% chance of nothing happening since P(c) = 0.90
        return a, b
    else:
        # converts floats to char list
        str_a = str(a)
        a_list = list(str_a)

        str_b = str(b)
        b_list = list(str_b)

        # stops problem of strings being two different sizes by adding zeros to the end of the list
        max_length = max(len(a_list), len(b_list))
        a_list += ['0'] * (max_length - len(a_list))
        b_list += ['0'] * (max_length - len(b_list))

        k = random.randint(1, len(a_list))  # creates a cross-over point
        # print("Crossover point :", k)

        # interchanging the genes
        for j in range(k, len(a_list)):
            a_list[j], b_list[j] = b_list[j], a_list[j]

        a_list = ''.join(a_list)
        b_list = ''.join(b_list)

        a_list, b_list = remove_dots(a_list, b_list)

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
        fit_values = population / pop_sum  # prob = f(x)/(sum of f(x) array) -> total of array will be close to "1"

    for i in range(15):  # 15 if half of N (N = 30)
        while True:
            par_a = roulette_wheel(population, fit_values)
            par_b = roulette_wheel(population, fit_values)

            if par_a != par_b:  # can't mate with itself
                break

        new_gen = np.append(new_gen, crossover(par_a, par_b))

        # p_mut = 0.05 => this means there is a 5% chance of mutation occurring
        mut_values = np.array([
            mutation(value) if random.random() < p_mut else value
            for value in new_gen
        ])

    print("First gen:")
    print(population)
    print("\nNew gen:")
    print(mut_values)
