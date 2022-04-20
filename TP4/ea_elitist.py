import numpy as np
from deap import base, creator, benchmarks
import random
from deap import tools
from scoop import futures

# <INIT_DEAP>
from deap import tools, algorithms


def set_creator(cr):
    global creator
    creator = cr


weights = (-1.0,)

from deap import creator

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


def ea_elitist(
    n, nbgen, evaluate, IND_SIZE, MIN_V=-5, MAX_V=5, CXPB=0.5, MUTPB=0.2, weights=(1,)
):
    """Algorithme evolutionniste avec sélection élitiste

    Algorithme evolutionniste avec sélection élitiste.
    :param n: taille de la population
    :param nbgen: nombre de generation
    :param evaluate: la fonction d'évaluation
    :param IND_SIZE: la taille d'un individu
    :param MIN_V: la valeur minimale d'un paramètre du génome
    :param MAX_V: la valeur maximale d'un paramètre du génome
    :param CXPB: la probabilité de croisement
    :param MUTPB: la probabilité de mutation
    :param weights: les poids à utiliser pour la somme pondérée en cas de problème multi-objectif (pour en faire une somme pondérée), ATTENTION, c'est différent du paramètre weights de DEAP qui détermine si c'est une maximisation ou une minimisation
    """

    # print("EA Elitist: n=%d nbgen=%d, IND_SIZE=%d, MIN_V=%f, MAX_V=%f, CXPB=%f, MUTPB=%f"%(n,nbgen,IND_SIZE, MIN_V, MAX_V, CXPB, MUTPB)+" weights="+str(weights))

    toolbox = base.Toolbox()

    toolbox.register("map", futures.map)

    # à compléter pour sélectionner les opérateurs de mutation, croisement, sélection avec des toolbox.register(...)
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
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    # </ANSWER>

    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()

    # La structure permettant de récupérer le meilleur individu
    hof = tools.HallOfFame(1)

    population = toolbox.population(n=n)

    # Evaluate the individuals with an invalid fitness
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    if len(weights) != len(fitnesses[0]):
        print(
            "ERROR: the weights and the fitness should have the same size ! weights="
            + str(weights)
            + " fitness="
            + str(fitnesses[0])
        )
        return
    spop = []
    for ind, fit in zip(population, fitnesses):
        # print("Fit: "+str(fit))
        prod = [weights[i] * fit[i] for i in range(len(fit))]
        ind.fitness.values = (sum(prod),)
        ind.lfit = fit
        spop.append(fit)
    hof.update(population)
    stat = stats.compile(population)
    logbook.record(gen=0, pop=spop, best=hof[0].lfit, **stat)
    # le champ "best" pourra être utilisé pour faciliter les comparaisons avec NSGA-2 dans l'expérience multi-objectif de contrôle du pendule (question 5.2)

    for g in range(1, nbgen):
        ## à compléter en n'oubliant pas de mettre à jour les statistiques, le logbook et le hall-of-fame comme cela a été fait pour la génération 0

        # <ANSWER>
        # Select the next generation individuals
        offspring = algorithms.varAnd(population, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            prod = [weights[i] * fit[i] for i in range(len(fit))]
            ind.fitness.values = (sum(prod),)
            ind.lfit = fit

        # The population is entirely replaced by the offspring
        population = toolbox.select(population + offspring, n)

        hof.update(population)
        stat = stats.compile(population)
        logbook.record(gen=g, best=hof[0].lfit, **stat)
        # </ANSWER>

    return population, hof, logbook
