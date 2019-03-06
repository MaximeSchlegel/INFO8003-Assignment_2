import random as rdm
from src.domain.HillDomain import HillDomain


class Policy:
    # Permet de créer des policy qui seront standart, la class possède des policy predefinies

    def __init__(self, fn=None):
        self.domain = HillDomain
        if (fn is None) or (fn == "random"):
            self.fn = self.random
        elif fn == "right":
            self.fn = self.right
        elif fn == "left":
            self.fn = self.left
        else:
            assert callable(fn)
            self.fn = fn

    def __call__(self, *args, **kwargs):
        # Permet à l'objet de se comporter comme une fonction, on s'assure que les arguments soit du bon type
        assert len(kwargs) == 0
        assert len(args) == 1
        assert isinstance(*args, tuple)
        assert len(*args) == 2
        return self.fn(*args)

    def random(self, position):
        return rdm.choice(self.domain.VALID_ACTIONS)

    def right(self, position):
        return self.domain.RIGHT

    def left(self, position):
        return self.domain.LEFT
