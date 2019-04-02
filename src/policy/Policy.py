from src.hillproblem.HillDomain import HillDomain
import imageio
import numpy
import random


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
        return random.choice(self.domain.VALID_ACTIONS)

    def right(self, position):
        return self.domain.RIGHT

    def left(self, position):
        return self.domain.LEFT

    def display(self):
        position_points = 40
        speed_point = 120
        m = numpy.array([[0. for _ in range(10 * position_points)] for _ in range(10 * speed_point)])
        for i in range(position_points):
            p = -1 + (i / 20)
            for j in range(speed_point):
                s = -3 + (j / 20)
                if self((s,p)) == -4:
                    c = 200.  # Black
                else:
                    c = 0. # White
                m[(10 * speed_point) - (10 * j) - 10: (10 * speed_point) - (10 * j), (10 * i): (10 * i) + 10] = numpy.full((10, 10), c)

        m[(10 * speed_point)//2 - 1: (10 * speed_point)//2 + 1, :] = numpy.full((2, 400), 255)
        m[:, (10 * position_points)//2 - 1: (10 * position_points)//2 + 1] = numpy.full((1200, 2), 255)
        imageio.imwrite('images/policy.png', m.astype("uint8"))
