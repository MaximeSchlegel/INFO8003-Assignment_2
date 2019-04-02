from src.hillproblem.HillDomain import HillDomain
from src.policy.Policy import Policy
from src.animator.Animator import Animator
from src.estimator.Estimator import Estimator

import imageio
import math
import matplotlib.pyplot as plt
import numpy
import random
import time


class Agent:

    def __init__(self, policy: Policy, estimator: Estimator, gamma=0.95):
        self.domain = HillDomain()
        self.estimator = estimator
        self.policy = policy
        self.gamma = gamma
        self.action_state_estimator = []
        self.history = []
        self.max_reward = 0

    def play(self, nb_actions=None, display=False, animation=False):
        if display:
            print("Game Started")
        initial_state = (-0.5, 0.)
        history_state = [initial_state]  # Hold the list of visited state used to diplay the party
        i = 0

        while not self.domain.ended(initial_state) and ((nb_actions is None) or (i < nb_actions)):
            # stop when the party is finish (lost or win) or if the user has set a limit when it is reach
            action = self.policy(initial_state)  # Chose an action according to the policy
            final_state, h = self.domain.move(initial_state, action, True)  # Compute the resulting state
            reward = self.domain.reward(initial_state, action) # Compute the reward
            if abs(reward) > abs(self.max_reward):
                self.max_reward = reward
            history_state += h
            self.history.append((initial_state, action, reward, final_state))
            initial_state = final_state
            i += 1

        if display:
            print('Recap :')
            print("  - Nb Actions : {}".format(i))
            print("  - Resultat : {}".format(self.history[-1][2]))

        if animation:
            t1 = time.time()
            anim = Animator()
            print("          ... Creating Animation ...")
            anim(history_state, dt=0.001)
            t2 = time.time()
            print("               Replay available\n"
                  "          (rendering duration : {:.2f})".format(t2-t1), '\n')

        return self.history[-1][2]

    def expected_return_compute(self, n=100, display=False):
        # Approximate the expected return of a policy by using the Montecarlo method
        cumulated_return = 0

        for _ in range(n):
            result = self.play()
            cumulated_return += result[-1][2]

        if display:
            print("Expected Return : {:.3f}".format(cumulated_return / n), '\n')

        return cumulated_return / n

    def action_state_compute_estimator(self, n, k=1):
        # Compute the estimator for the Q fonction of rank n
        if len(self.action_state_estimator) > n:
            return self.action_state_estimator[n]

        print('Training New Estimators: ')
        if len(self.history) == 0:
            raise Exception("Buffer is empty")
        print("  Buffer size : {}".format(len(self.history)))

        for it in range(len(self.action_state_estimator), n + 1):
            print ("  Training Q{}".format(it))
            for _ in range(k):
                ti = [[osst[0], osst[1]] for osst in self.history]
                if it == 0:
                    to = numpy.array([osst[2] for osst in self.history])
                else:
                    to = []
                    for i in range(len(self.history)):
                        r = self.history[i][2]
                        m = max(self.action_state_estimator[-1](self.history[i][3], self.domain.LEFT),
                                self.action_state_estimator[-1](self.history[i][3], self.domain.RIGHT))
                        to.append(r + self.gamma * m)
                # print(ti, '\n', to)
                estimator = self.estimator()
                estimator.train(ti, numpy.array(to))
            self.action_state_estimator.append(estimator)
            print('    Done')

        print('Sucess')
        return self.action_state_estimator[-1]

    def action_state_error(self, error):
        return math.log((error * math.pow(1 - self.gamma), 2) / (2 * self.max_reward)) / math.log(self.gamma)

    def action_state_approxiamate(self, error):
        self.action_state_compute_estimator(self.action_state_error(error))

    def policy_set_optimal(self):
        def best(state):
            left = self.action_state_estimator[-1](state, self.domain.LEFT)
            right = self.action_state_estimator[-1](state, self.domain.RIGHT)
            if left > right:
                return self.domain.LEFT
            elif right > left:
                return self.domain.RIGHT
            return random.choice([self.domain.RIGHT, self.domain.LEFT])
        self.policy = Policy(best)

    def action_state_display(self, n):
        if n == -1:
            n = len(self.action_state_estimator) - 1
        x = numpy.linspace(-1, 1, 200)
        y = numpy.linspace(-3, 3, 200)
        qn = self.action_state_compute_estimator(n)
        vector_img_right = numpy.zeros((200, 200))
        vector_img_left = numpy.zeros((200, 200))
        vector_img = numpy.zeros((200, 200))

        for i in range(200):
            for j in range(200):
                vector_img_right[199 - j, i] = qn((x[i], y[j]), 4)
                vector_img_left[199 - j, i] = qn((x[i], y[j]), -4)
                if vector_img_left[199 - j, i] <= vector_img_right[199 - j, i]:
                    vector_img[199 - j, i] = 4
                else:
                    vector_img[199 - j, i] = -4

        plt.figure(1)
        cs = plt.contourf(x, y, vector_img_right, cmap='Spectral')
        plt.colorbar(cs)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.title("Evalutation for the right action for Q{}".format(n))
        # plt.show()
        # plt.clf()

        plt.figure(2)
        cs2 = plt.contourf(x, y, vector_img_left, cmap='Spectral')
        plt.colorbar(cs2)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.title("Evalutation for the left action for Q{}".format(n))
        plt.title
        # plt.show()
        # plt.clf()

        plt.figure(3)
        cs3 = plt.contourf(x, y, vector_img, cmap='Spectral')
        plt.colorbar(cs3)
        plt.xlabel("position")
        plt.ylabel("speed")
        plt.title("Representation of Q{}".format(n))
        plt.show()
