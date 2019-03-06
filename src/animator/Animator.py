import imageio
import numpy as np
import math as m
import matplotlib.pyplot as plt
from PIL import Image
from src.hillproblem.HillDomain import HillDomain


class Animator:

    def __init__(self, size=50):
        self.domain = HillDomain()
        self.width, self.height = 16 * size, 9 * size
        self.x_max, self.x_min = 1.1, -1.1
        self.y_max, self.y_min = self.domain.hill(1.1), self.domain.hill(-0.5)
        self.s_max = 3
        self.x_factor = self.width / (self.x_max - self.x_min)
        self.y_scale = 0.9
        self.y_factor = self.y_scale * self.height / (self.y_min - self.y_max)
        self.y_base = 2 * self.height * (1 - self.y_scale) / 3
        self.background = None
        self.create_background()

    def xtow(self, x):
        return int((x - self.x_min) * self.x_factor)

    def wtox(self, w):
        return self.x_min + w / self.x_factor

    def ytoh(self, y):
        return int(self.y_base + (y - self.y_max) * self.y_factor)

    def htoy(self, h):
        return self.y_max + ((h - self.y_base) / self.y_factor)

    def create_background(self):
        try:
            img = Image.open('src/animator/background.jpeg')
            self.background = np.array(img)
        except:
            pics = Image.open("src/animator/pine_tree.png")
            tree = np.delete(np.array(pics), 3, 2)
            tree_mask = np.where(tree != 0, 1, 0)
            self.background = np.zeros((self.height, self.width, 3))
            for w in range(self.width):
                for h in range(self.height):
                    if h - 1 > self.ytoh(self.domain.hill(self.wtox(w))):
                        # colline
                        self.background[h][w] = [78, 155, 69]
                    elif h + 2 < self.ytoh(self.domain.hill(self.wtox(w))):
                        # ciel
                        self.background[h][w] = [24, 113, 206]

            w_pos = self.xtow(-0.5)
            h_pos = self.ytoh(self.domain.hill(-0.5))
            self.background[h_pos - 42: h_pos, w_pos - 23: w_pos + 24] += tree_mask * (tree - self.background[h_pos - 42: h_pos, w_pos - 23: w_pos + 24])
            w_pos = self.xtow(0.5)
            h_pos = self.ytoh(self.domain.hill(0.5))
            self.background[h_pos - 42: h_pos, w_pos - 23: w_pos + 24] += tree_mask * (tree - self.background[h_pos - 42: h_pos, w_pos - 23: w_pos + 24])
            img = Image.fromarray(self.background.astype('uint8'), mode='RGB')
            img.save('src/animator/background.jpeg')
        return self.background

    def draw_car(self, img, x):
        w_pos = self.xtow(x)
        h_pos = self.ytoh(self.domain.hill(x))
        r = m.degrees(m.atan(self.domain.hill_d1(x)))
        pics = Image.open('src/animator/car.png')
        pics = pics.rotate(r, expand=True, center=(24, 7))
        car = np.delete(np.array(pics), 3, 2)
        car_mask = np.where(car != 0, 1, 0)
        dy, dx, p = car.shape
        if dy % 2 != 0:
            dy -= 1
            car = np.delete(car, dy, 0)
            car_mask = np.delete(car_mask, dy, 0)
        if dx % 2 != 0:
            dx -= 1
            car = np.delete(car, dx, 1)
            car_mask = np.delete(car_mask, dx, 1)
        dh_high = min(dy//2, self.height - h_pos)
        dh_low = max(-dy//2, -h_pos)
        dw_high = min(dx//2, self.width - w_pos)
        dw_low = max(-dx//2, -w_pos)
        img[h_pos + dh_low: h_pos + dh_high, w_pos + dw_low: w_pos + dw_high] += \
            car_mask[(dy//2) + dh_low: (dy//2) + dh_high, (dx//2) + dw_low: (dx//2) + dw_high].astype('uint8') * \
            (car[(dy//2) + dh_low: (dy//2) + dh_high, (dx//2) + dw_low: (dx//2) + dw_high].astype('uint8') -
             img[h_pos + dh_low: h_pos + dh_high, w_pos + dw_low: w_pos + dw_high])

    def draw_speed(self, img, s):
        h_s = int(abs(s) * 150 / self.s_max)
        w_pos = 740
        h_pos = 250
        if s > 0:
            img[h_pos - h_s:h_pos, w_pos + 5: w_pos + 35, 0:] = np.full((abs(h_s), 30, 1), 12)
            img[h_pos - h_s:h_pos, w_pos + 5: w_pos + 35, 1:2] = np.full((abs(h_s), 30, 1), 240)
            img[h_pos - h_s:h_pos, w_pos + 5: w_pos + 35, 2:3] = np.full((abs(h_s), 30, 1), 53)
        else:
            img[h_pos:h_pos + h_s, w_pos + 5: w_pos + 35, 0:] = np.full((abs(h_s), 30, 1), 250)
            img[h_pos:h_pos + h_s, w_pos + 5: w_pos + 35, 1:2] = np.full((abs(h_s), 30, 1), 49)
            img[h_pos:h_pos + h_s, w_pos + 5: w_pos + 35, 2:3] = np.full((abs(h_s), 30, 1), 12)
        img[h_pos - 3: h_pos+3, w_pos: w_pos + 40] = np.ones((6, 40, 3))

    def __call__(self, *args, **kwargs):
        history = args[0]
        figure = plt.figure()
        figure.set_size_inches(self.width / 100, self.height / 100)
        axes = figure.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        axes.set_xticks([])
        axes.set_yticks([])
        images = []
        # on crée notre figure, on l'ajuste à la bonne taille
        # on lui enleve les axes et autres composantes pour les graphiques
        image = np.copy(self.background)
        self.draw_car(image, history[0][0])
        self.draw_speed(image, history[0][1])
        plt_img = plt.imshow((image / 255), animated=True)
        plt_txt = plt.text(10, 440, 't = 0', color='black')
        images.append([plt_img, plt_txt])
        # on initialise notre animation avec l'image à t=0
        # on prend des états avec un pas de 20 (5 états par actions)
        # on passe 50 images par seconde => vitesse "réelle"
        s_step = 20
        aps = 10
        with imageio.get_writer('animation.gif', mode='I', fps=((100//s_step)*aps)) as writer:
            for i in range(1, (len(history) // s_step)):
                image = np.copy(self.background)
                self.draw_car(image, history[s_step * i][0])
                self.draw_speed(image, history[s_step * i][1])
                writer.append_data(image)
            for i in range((100//s_step)*aps):
                writer.append_data(image)
