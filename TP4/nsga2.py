import numpy as np

# <INIT_DEAP>
# Code d'initialisation du "créateur" de DEAP, partie à mettre dans un module
# NE PAS MODIFIER
def set_creator(cr):
    global creator
    creator = cr


from deap import base, tools, algorithms

weights = (-1.0, -1.0)

from deap import creator, base

set_creator(creator)

if hasattr(creator, "MaFitness"):
    # Deleting any previous definition (to avoid warning message)
    del creator.MaFitness
creator.create("MaFitness", base.Fitness, weights=(weights))

if hasattr(creator, "Individual"):
    # Deleting any previous definition (to avoid warning message)
    del creator.Individual
creator.create("Individual", list, fitness=creator.MaFitness)
# </INIT_DEAP>

from deap.tools._hypervolume import hv

from scoop import futures

import random

# ne pas oublier d'initialiser la grane aléatoire (le mieux étant de le faire dans le main))
random.seed()


def nsga2(
    n,
    nbgen,
    evaluate,
    IND_SIZE=5,
    ref_point=np.array([1, 1]),
    MIN_V=-5,
    MAX_V=5,
    CXPB=0.5,
    MUTPB=0.2,
):
    """NSGA-2

    NSGA-2
    :param n: taille de la population
    :param nbgen: nombre de generation
    :param evaluate: la fonction d'évaluation
    :param IND_SIZE: la taille d'un individu
    :param ref_point: le point de référence pour le calcul de l'hypervolume
    :param MIN_V: la valeur minimale d'un paramètre du génome
    :param MAX_V: la valeur maximale d'un paramètre du génome
    :param CXPB: la probabilité de croisement
    :param MUTPB: la probabilité de mutation
    """

    toolbox = base.Toolbox()
    paretofront = tools.ParetoFront()

    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()

    # à compléter
    # <ANSWER>
    toolbox.register("attribute", random.uniform, MIN_V, MAX_V)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=IND_SIZE,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxSimulatedBinary, eta=15.0)
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        eta=15.0,
        low=MIN_V,
        up=MAX_V,
        indpb=1 / IND_SIZE,
    )
    toolbox.register("select_dcd", tools.selTournamentDCD, k=n)
    toolbox.register("select", tools.selNSGA2, k=n, nd="standard")
    toolbox.register("evaluate", evaluate)
    # </ANSWER>

    toolbox.register("map", futures.map)

    population = toolbox.population(n=n)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        # print("Fit: "+str(fit))
        ind.fitness.values = fit

    paretofront.update(population)

    pffit = []
    for i in paretofront:
        pffit.append(i.fitness.values)

    pointset = [np.array(ind.fitness.getValues()) for ind in paretofront]
    shv = hv.hypervolume(pointset, ref_point)
    stat = stats.compile(population)
    logbook.record(gen=0, best=pffit, hypervolume=shv, **stat)
    # Le champ best contient la fitness de chaque individu du front de pareto, cela permettra de faciliter les comparaisons pour la question 5.2.

    # Begin the generational process
    population = toolbox.select(population)
    for gen in range(1, nbgen):
        #     if (gen%10==0):
        #         print("+",end="", flush=True)
        #     else:
        #         print(".",end="", flush=True)

        ## à compléter en n'oubliant pas de mettre à jour les statistiques, le logbook et le hall-of-fame comme cela a été fait pour la génération 0

        # <ANSWER>
        offspring = toolbox.select_dcd(population)
        offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring)
        paretofront.update(population)

        pointset = [np.array(ind.fitness.getValues()) for ind in paretofront]
        shv = hv.hypervolume(pointset, ref_point)
        stat = stats.compile(population)
        logbook.record(gen=gen, best=pffit, hypervolume=shv, **stat)
        # </ANSWER>

    return population, paretofront, logbook
