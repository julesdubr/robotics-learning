weights = (-1.0, -1.0)

from deap import creator, base
from nsga2 import set_creator, nsga2

set_creator(creator)
if hasattr(creator, "MaFitness"):
    # Deleting any previous definition (to avoid warning message)
    del creator.MaFitness
creator.create("MaFitness", base.Fitness, weights=(weights))

if hasattr(creator, "Individual"):
    # Deleting any previous definition (to avoid warning message)
    del creator.Individual
creator.create("Individual", list, fitness=creator.MaFitness)

from nn import SimpleNeuralControllerNumpy
import numpy as np
import matplotlib.pyplot as plt

import datetime
import pickle
import os

import gym


nn = SimpleNeuralControllerNumpy(4, 1, 2, 5)
IND_SIZE = len(nn.get_parameters())
env = gym.make("CartPole-v1")


def eval_nn(genotype, nbeval=1, render=False, nbstep=1000):
    """Evaluation d'une politique parametrée par le génotype

    Evaluation d'une politique parametrée par le génotype
    :param genotype: le paramètre de politique à évaluer
    :param nbeval: le nombre de répétitions de l'évaluation à réaliser (une évaluation commençant à une position aléatoire)
    :param render: affichage du pendule
    :param nbstep: durée maximale d'une évaluation
    """
    ## à completer

    # ATTENTION: si le pendule tombe (done), vous pouvez interrompre l'évaluation pour accélérer les calculs,
    # mais il faut compléter le calcul d'erreur pour ne pas favoriser ceux qui tombent rapidement...

    # La politique sera être créée avec l'instruction suivante:
    # nn=nn.SimpleNeuralControllerNumpy(4,1,2,5).
    # Vous utiliserez la fonction set_parameters pour positionner ses paramètres à partir du génotype
    # et predict pour calculer l'action suggérée par la politique

    # Le pendule a comme action 0 ou 1 qui correspondra à une force maximale dans un sens ou dans l'autre.
    # L'action sera 0 pour une valeur de sortie négative du réseau de neurones et 1 sinon.

    total_x = 0
    total_theta = 0
    nn.set_parameters(genotype)

    for _ in range(nbeval):
        observation = env.reset()

        for step in range(nbstep):
            if render:
                env.render()

            action = int(nn.predict(observation)[0] > 0)
            observation, _, done, _ = env.step(action)

            if done:
                total_x += abs(observation[0]) * (nbstep - step)
                total_theta += abs(observation[2]) * nbstep - step
                break

            total_x += abs(observation[0])
            total_theta += abs(observation[2])

    return (total_x, total_theta)


if __name__ == "__main__":
    name = "nsga2_cartpole"
    d = datetime.datetime.today()
    dir = d.strftime(name + "_%Y_%m_%d-%H-%M-%S")
    os.mkdir(dir)

    # with open()
    print("Results in " + dir)

    # <ANSWER>
    ref_point = [1000, 1000]
    print

    plt.figure()
    plt.title("Hypervolume mean vs iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Hypervolume mean")

    nbsteps = 10

    for pop_size in [4, 8, 64, 128]:
        data = []
        for i in range(10):
            _, hof, logbook = nsga2(pop_size, nbsteps, eval_nn, IND_SIZE, ref_point)
            data.append(logbook)

            subdir = dir + f"/pop_{pop_size}"
            if i == 0:
                os.mkdir(subdir)
                os.mkdir(subdir + "/logbooks")
                os.mkdir(subdir + "/hofs")

            with open(subdir + f"/logbooks/logbook_{i}.pkl", "wb") as f:
                pickle.dump(logbook, f)
            with open(subdir + f"/hofs/hof_{i}.pkl", "wb") as f:
                pickle.dump(hof, f)

        gen = data[0].select("gen")

        genscore = [
            [np.mean(data[j].select("hypervolume")[i]) for j in range(10)]
            for i in range(nbsteps)
        ]
        median = [np.median(genscore[i]) for i in range(nbsteps)]
        fit_25 = [np.quantile(genscore[i], 0.25) for i in range(nbsteps)]
        fit_75 = [np.quantile(genscore[i], 0.75) for i in range(nbsteps)]
        plt.plot(gen, median, label=f"pop={pop_size}")
        plt.fill_between(gen, fit_25, fit_75, alpha=0.25, linewidth=0)

        with open(subdir + f"/data.pkl", "wb") as f:
            data = {
                "gen": gen,
                "median": median,
                "fit_25": fit_25,
                "fit_75": fit_75,
            }
            pickle.dump(data, f)

    env.close()
    plt.legend()
    plt.show()
    # </ANSWER>
