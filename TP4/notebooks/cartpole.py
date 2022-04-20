# A COMPLETER
# Squelette du code de réponse à la question 4. 

import array
import random

import math
import gym


from nn import SimpleNeuralControllerNumpy

import datetime
import pickle



# Pour récupérer le nombre de paramètre. voir fixed_structure_nn_numpy pour la signification des paramètres. Le TME fonctionne avec ces paramètres là, mais vous pouvez explorer des valeurs différentes si vous le souhaitez.
nn=SimpleNeuralControllerNumpy(4,1,2,5)
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

    
    total_x=0
    total_theta=0
    #<ANSWER>

    #</ANSWER>
    return (total_x, total_theta)
