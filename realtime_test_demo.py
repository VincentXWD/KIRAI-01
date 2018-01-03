import pygame
from pygame.locals import *
import math
import cv2
from hpelm import ELM
import read_mnist
import hog_descriptor
import numpy as np

IMG_PATH = './resource/test_num.png'
elm = None

class Brush():
  def __init__(self, screen):
    self.screen = screen
    self.color = (255, 255, 255)
    self.size = 1
    self.drawing = False
    self.last_pos = None
    self.space = 1
    self.style = False
    self.brush = pygame.image.load("./resource/brush.png").convert_alpha()
    self.brush_now = self.brush.subsurface((0, 0), (1, 1))


  def start_draw(self, pos):
    self.drawing = True
    self.last_pos = pos


  def end_draw(self):
    self.drawing = False


  def set_brush_style(self, style):
    print "* set brush style to", style
    self.style = style


  def get_brush_style(self):
    return self.style


  def set_size(self, size):
    if size < 0.5:
      size = 0.5
    elif size > 50:
      size = 50
    print "* set brush size to", size
    self.size = size
    self.brush_now = self.brush.subsurface((0, 0), (size * 2, size * 2))


  def get_size(self):
    return self.size


  def draw(self, pos):
    if self.drawing:
      for p in self._get_points(pos):
        if self.style == False:
          pygame.draw.circle(self.screen,
                             self.color, p, self.size)
        else:
          self.screen.blit(self.brush_now, p)

      self.last_pos = pos


  def _get_points(self, pos):
    points = [(self.last_pos[0], self.last_pos[1])]
    len_x = pos[0] - self.last_pos[0]
    len_y = pos[1] - self.last_pos[1]
    length = math.sqrt(len_x ** 2 + len_y ** 2)
    step_x = len_x / length
    step_y = len_y / length
    for i in xrange(int(length)):
      points.append(
        (points[-1][0] + step_x, points[-1][1] + step_y))
    points = map(lambda x: (int(0.5 + x[0]), int(0.5 + x[1])), points)
    return list(set(points))

class Painter():
  def __init__(self):
    self.screen = pygame.display.set_mode((200, 200))
    pygame.display.set_caption("Painter")
    self.clock = pygame.time.Clock()
    self.brush = Brush(self.screen)


  def run(self):
    self.screen.fill((0,0,0))
    while True:
      self.clock.tick(500)
      for event in pygame.event.get():
        if event.type == QUIT:
          return
        elif event.type == KEYDOWN:
          if event.key == K_c:
            self.screen.fill((0,0,0))
          elif event.key == K_s:
            pygame.image.save(self.screen, IMG_PATH)
            print 'Image saved. now predicting: '
            print 'Well, I guess the number is :', predict(IMG_PATH)

        elif event.type == MOUSEBUTTONDOWN:
          self.brush.start_draw(event.pos)
        elif event.type == MOUSEMOTION:
          self.brush.draw(event.pos)
        elif event.type == MOUSEBUTTONUP:
          self.brush.end_draw()
      pygame.display.update()


def predictor_init(model_path):
  elm = ELM(324, 324)
  elm.load(model_path)
  return elm


def get_image(img_path):
  image = cv2.imread(img_path)
  image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
  _, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
  return image


def get_labels(result):
  ret, id = 0, 0
  for i in range(0, len(result)):
    if ret < result[i]:
      id, ret = i, result[i]
  return id


def predict(img_path):
  image = get_image(img_path)
  # cv2.imwrite('test.png', image)
  img_hog = hog_descriptor.get_hog(image)
  img_hog = np.array(img_hog).T

  return get_labels(elm.predict(img_hog)[0])


if __name__ == '__main__':
  elm = predictor_init('./elm.model')
  app = Painter()
  app.run()
