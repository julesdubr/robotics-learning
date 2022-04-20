weights=(-1.0,)

from deap import creator, base
from ea_elitist import set_creator,ea_elitist

set_creator(creator)
if (hasattr(creator, "MaFitness")):
    # Deleting any previous definition (to avoid warning message)
    del creator.MaFitness
creator.create("MaFitness", base.Fitness, weights=(weights))

if (hasattr(creator, "Individual")):
    # Deleting any previous definition (to avoid warning message)
    del creator.Individual
creator.create("Individual", list, fitness=creator.MaFitness)


from nn import SimpleNeuralControllerNumpy
import numpy as np
import matplotlib.pyplot as plt

import datetime
import pickle
import os

import array
import random

import math
import gym


nn = SimpleNeuralControllerNumpy(4,1,2,5)
IND_SIZE=len(nn.get_parameters())
env = gym.make('CartPole-v1')

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
    
    total_reward=0
    nn.set_parameters(genotype)

    observation = env.reset()

    done = False
    epoch = 0
    while epoch < nbstep and not done:
        action = int(nn.predict(observation)[0] > 0)
        observation, reward, done, info = env.step(action)
        total_reward = reward

        total_reward += reward

        epoch += 1

    total_reward += nbstep-epoch
    return (nbstep - total_reward,)


if __name__ == '__main__':

    name="elitist_cartpole"
    d=datetime.datetime.today()
    dir=d.strftime(name+"_%Y_%m_%d-%H-%M-%S")
    os.mkdir(dir)
    
    print("Results in "+dir)

    # à compléter pour faire appel à un algorithme évolutionniste mono-objectif sur le cartpole.
    # Il est suggéré de sauvegarder le ou les logbook et hof dans le répertoire 'dir'
    #<ANSWER>
    plt.figure()
    plt.title("Fitness mean vs iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness mean")

    nbsteps = 10

    for pop_size in [5, 10, 100, 200]:
        data = []
        for _ in range(10):
            data.append(ea_elitist(pop_size, 10, eval_nn, IND_SIZE)[2])

        gen = data[0].select("gen")

        genscore = [[data[j].select("avg")[i] for j in range(10)] for i in range(10)]
        median = [np.median(genscore[i]) for i in range(10)]
        fit_25 = [np.quantile(genscore[i],0.25) for i in range(10)]
        fit_75 = [np.quantile(genscore[i],0.75) for i in range(10)]
        plt.plot(gen, median, label=f"pop={pop_size}")
        plt.fill_between(gen, fit_25, fit_75, alpha=0.25, linewidth=0)

    env.close()
    plt.legend()
    plt.show()
    #</ANSWER>