import math as m


class HillDomain:

    RIGHT = 4
    LEFT = -4
    VALID_ACTIONS = [RIGHT, LEFT]

    def __init__(self):
        self.dt_action = 0.1
        self.dt_integration = 0.001
        self.m = 1
        self.g = 9.81

    def hill(self, p):
        # y en fonction de la position
        if p < 0:
            return m.pow(p, 2) + p
        else:
            return p / (m.sqrt(1 + 5 * m.pow(p, 2)))

    def hill_d1(self, p):
        # derivee premiere
        if p < 0:
            return 2 * p + 1
        else:
            return 1 / m.pow((1 + 5 * m.pow(p, 2)), 3 / 2)

    def hill_d2(self, p):
        # derivee seconde
        if p < 0:
            return 2
        else:
            return (-15 * p) / m.pow((1 + 5 * m.pow(p, 2)), 5 / 2)

    def dynamics(self, state, action):
        # systeme d'equations qui donne la nouvelle position/vitesse en fonction des anciennes et de l'action choisie
        position, speed = state
        dspeed = action / (self.m * (1 + m.pow(self.hill_d1(position), 2)))
        dspeed -= (self.g * self.hill_d1(position)) / (1 + m.pow(self.hill_d1(position), 2))
        dspeed -= (m.pow(speed, 2) * self.hill_d1(position) * self.hill_d2(position)) / \
                  (1 + pow(self.hill_d1(position), 2))
        dposition = speed
        return dposition, dspeed

    def move(self, state, action, history=False):
        # calcule la position suivante dans laquelle l'agent devra choisir une nouvelle action
        history_state = [state]
        for i in range(int(self.dt_action // self.dt_integration)):
            dposition, dspeed = self.dynamics(state, action)
            position = state[0] + (dposition * self.dt_integration)
            speed = state[1] + (dspeed * self.dt_integration)
            state = position, speed
            history_state.append(state)
        if history:
            return state, history_state  # history peut etre recupere pour display la solution
        return state

    def ended(self, state):
        # indique si la position de l'agent est terminale
        position, speed = state
        if (-1 < position < 1) and (-3 < speed < 3):
            return False
        return True

    def reward(self, state, action):
        # donne la reward de l'agent
        position, speed = self.move(state, action, False)
        if position < -1 or abs(speed) > 3:
            return -1
        if position > 1 and speed <= 3:
            return 1
        return 0
