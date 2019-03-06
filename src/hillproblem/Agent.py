from src.hillproblem.HillDomain import HillDomain
from src.policy.Policy import Policy
from src.animator.Animator import Animator
import time


class Agent:

    def __init__(self, policy: Policy, gamma=0.4):
        self.domain = HillDomain()
        self.policy = policy
        self.gamma = gamma

    def play(self, nb_actions=None, display=False, animation=False):
        initial_state = (-0.5, 0.)
        history_state = [initial_state]
        # Contient unique la liste des états visités, utilisée pour le display de la solution
        history_one_step_system_transition = []
        # Contient les one step system transition qui serviront pour l'apprentissage
        i = 0
        while not self.domain.ended(initial_state) and ((nb_actions is None) or (i < nb_actions)):
            # on s'arrete soit quand on arrive dans un etat terminal, soit qaund on atteind la limite fixée par
            # l'utilsateur (utile quand on veut display uniquement le début de certaine policy
            action = self.policy(initial_state)
            final_state, h = self.domain.move(initial_state, action, True)
            # On recupere toujours l'historique pour le renvoyer si après on veut le retraiter et le display
            reward = self.domain.reward(initial_state, action)
            history_state += h
            history_one_step_system_transition.append((initial_state, action, reward, final_state))
            initial_state = final_state
            i += 1
        if display:
            print('Recap :')
            print("  - Nb Actions : {}".format(len(history_one_step_system_transition)))
            print("  - Resultat : {}".format(history_one_step_system_transition[-1][2]))
        if animation:
            t1 = time.time()
            anim = Animator()
            print("          ... Creating Animation ...")
            anim(history_state, dt=0.001)
            t2 = time.time()
            print("               Replay available\n"
                  "          (rendering duration : {:.2f})".format(t2-t1), '\n')
        return history_one_step_system_transition

    def expected_return_compute(self, n=100, display=False):
        # Calcule une approximation de l'expected reward par une succession d'experience
        # erreur en n^(-1/2)
        cumulated_return = 0
        for _ in range(n):
            hosst = self.play()
            cumulated_return += hosst[-1][2]
        if display:
            print("Expected Return : {:.3f}".format(cumulated_return / n), '\n')
        return cumulated_return / n
