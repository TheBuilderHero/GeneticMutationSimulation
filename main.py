import numpy as np
import random


def f_equation(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2  # custom-provided function for assigning fitness


def mutation(mut_children):  # uses gene-wise mutation
    for row in mut_children:
        if random.random() < 0.05:  # mutations have a 5% chance of occurring
            gene_index = np.random.choice(len(row))
            selected_gene = row[gene_index]
            new_gene = random.uniform(selected_gene - 1, selected_gene + 1)  # creates range for new mutant gene
            row[gene_index] = new_gene                                       # replaced with new gene based on indices


def crossover(pairs):
    child = np.empty((0, 3), dtype=float)  # Initialize an empty array for results

    index = random.choice([1, 2])

    for pair in pairs:
        ind_1, ind_2 = pair  # create individuals 1 & 2 based on pairs

        if random.random() < 0.1:  # 10% chance that crossover is skipped (P(c) = 90%)
            # add values if they're skipped
            child = np.vstack((child, ind_1))
            child = np.vstack((child, ind_2))
            continue

        # print("Individual 1:", ind_1)
        # print("Individual 2:", ind_2)

        temp = ind_1[index].copy()  # swaps third column
        ind_1[index] = ind_2[index]
        ind_2[index] = temp

        if index == 1:  # swaps second column
            temp_2 = ind_1[index + 1].copy()
            ind_1[index + 1] = ind_2[index + 1]
            ind_2[index + 1] = temp_2

            child = np.vstack((child, ind_1))
            child = np.vstack((child, ind_2))
        else:
            # add values even if only third column is swapped
            child = np.vstack((child, ind_1))
            child = np.vstack((child, ind_2))

    return child


"""https://gist.github.com/rocreguant/e9f2481f4e9842dd76e9c61f653eb7c0
   adapted from above source for individuals instead of chromosomes"""
def proportional_selection(population, fitness_scores):
    total_fitness = sum(fitness_scores)

    # PSEUDOCODE: prob = f(x)/(sum of f(x) array) -> total of array will be close to "1"
    prob = [fitness / total_fitness for fitness in fitness_scores]

    pairs = []  # empty array to store 15 pairs (N = 30)
    for i in range(15):
        # "weights=prob" means individuals with higher weights are more often to be picked
        indices = random.choices(range(len(population)), weights=prob, k=2)  # k=2 creates pairs of 2
        pairs.append([population[i] for i in indices])

    return pairs


if __name__ == "__main__":
    gen = 2         # used for tracking generation number

    genes = np.random.uniform(-4, 5, size=(30, 3))         # initializes array of random values -4 <= x <= 5
    print(f"Gen {gen - 1}:\n", genes)                           # initial population, N = 30

    # generate fitness numbers
    fit_score = (np.apply_along_axis(f_equation, 1, genes))  # flip sign for maximization/minimization
    print("Fitness Scores 2:", genes)
    selection_pairs = proportional_selection(genes, fit_score)  # use proportional selection
    children = crossover(selection_pairs)                       # crossover selection
    mutation(children)                                          # mutation (5% chance)

    print(f"Gen {gen}:\n", children)
    print("Fitness Score 2:", fit_score)

    for i in range(48):
        gen = gen + 1
        print(f"Gen {gen}:\n", children)

        fit_score = np.apply_along_axis(f_equation, 1, children)
        selection_pairs = proportional_selection(children, fit_score)
        children = crossover(selection_pairs)
        mutation(children)

        print(f"Fitness Score {gen}:\n", fit_score)

