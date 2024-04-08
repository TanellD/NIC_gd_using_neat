import csv
import os
import random
import sys

# import the pygame module
import pygame
import neat

# will make it easier to use pygame functions
from pygame.math import Vector2
from pygame.draw import rect

# initializes the pygame module
pygame.init()

# creates a screen variable of size 800 x 600
screen = pygame.display.set_mode([800, 600])

# controls the main game while loop
done = False

# controls whether or not to start the game from the main menu
start = False

# sets the frame rate of the program
clock = pygame.time.Clock()

"""
CONSTANTS
"""
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

color = lambda: tuple([random.randint(0, 255) for i in range(3)])  # lambda function for random color, not a constant.
GRAVITY = Vector2(0, 0.86)  # Vector2 is a pygame

"""
Main player class
"""


class Player(pygame.sprite.Sprite):
    """Class for player. Holds update method, win and die variables, collisions and more."""
    win: bool
    died: bool

    def __init__(self, image, platforms, pos, *groups):
        """
        :param image: block face avatar
        :param platforms: obstacles such as coins, blocks, spikes, and orbs
        :param pos: starting position
        :param groups: takes any number of sprite groups.
        """
        super().__init__(*groups)
        self.onGround = False  # player on ground?
        self.platforms = platforms  # obstacles but create a class variable for it
        self.died = False  # player died?
        self.win = False  # player beat level?

        self.image = pygame.transform.smoothscale(image, (32, 32))
        self.rect = self.image.get_rect(center=pos)  # get rect gets a Rect object from the image
        self.jump_amount = 10.9  # jump strength
        self.particles = []  # player trail
        self.isjump = False  # is the player jumping?
        self.vel = Vector2(0, 0)  # velocity starts at zero

    def collide(self, yvel, platforms):
        for p in platforms:
            if pygame.sprite.collide_rect(self, p):
                """pygame sprite builtin collision method,
                sees if player is colliding with any obstacles"""
                if isinstance(p, Orb):
                    pygame.draw.circle(alpha_surf, (255, 255, 0), p.rect.center, 18)
                    screen.blit(pygame.image.load("images/editor-0.9s-47px.gif"), p.rect.center)
                    self.jump_amount = 12  # gives a little boost when hit orb
                    self.jump()
                    self.jump_amount = 10.9  # return jump_amount to normal

                elif isinstance(p, End):
                    self.win = True

                elif isinstance(p, Spike):
                    self.died = True  # die on spike
                elif isinstance(p, Platform):  # these are the blocks (may be confusing due to self.platforms)
                    if yvel > 0:
                        """if player is going down(yvel is +)"""
                        self.rect.bottom = p.rect.top  # dont let the player go through the ground
                        self.vel.y = 0  # rest y velocity because player is on ground

                        # set self.onGround to true because player collided with the ground
                        self.onGround = True

                        # reset jump
                        self.isjump = False
                    elif yvel < 0:
                        """if yvel is (-),player collided while jumping"""
                        self.rect.top = p.rect.bottom  # player top is set the bottom of block like it hits it head

                    else:
                        """otherwise, if player collides with a block, he/she dies."""
                        self.vel.x = 0
                        self.rect.right = p.rect.left  # dont let player go through walls
                        self.died = True

    def jump(self):
        self.vel.y = -self.jump_amount  # players vertical velocity is negative so ^

    def update(self):
        """update player"""
        global elements
        if self.isjump:
            if self.onGround:
                """if player wants to jump and player is on the ground: only then is jump allowed"""
                self.jump()

        if not self.onGround:  # only accelerate with gravity if in the air
            self.vel += GRAVITY  # Gravity falls

            # max falling speed
            if self.vel.y > 100: self.vel.y = 100

        # do x-axis collisions
        self.collide(0, self.platforms)

        # increment in y direction
        self.rect.top += self.vel.y

        # assuming player in the air, and if not it will be set to inversed after collide
        self.onGround = False

        # do y-axis collisions
        self.collide(self.vel.y, self.platforms)

        # check if we won or if player won
        # eval_outcome(self.win, self.died)


"""
Obstacle classes
"""


# Parent class
class Draw(pygame.sprite.Sprite):
    """parent class to all obstacle classes; Sprite class"""

    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image
        self.rect = self.image.get_rect(topleft=pos)


#  ====================================================================================================================#
#  classes of all obstacles. this may seem repetitive but it is useful(to my knowledge)
#  ====================================================================================================================#
# children
class Platform(Draw):
    """block"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Spike(Draw):
    """spike"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class Orb(Draw):
    """orb. click space or up arrow while on it to jump in midair"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


class End(Draw):
    """place this at the end of the level"""

    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


"""
Functions
"""


def init_level(map):
    """this is similar to 2d lists. it goes through a list of lists, and creates instances of certain obstacles
    depending on the item in the list"""
    x = 0
    y = 0

    for row in map:
        for col in row:

            if col == "0":
                Platform(block, (x, y), elements)

            if col == "Spike":
                Spike(spike, (x, y), elements)
            if col == "Orb":
                orbs.append([x, y])
                Orb(orb, (x, y), elements)
            if col == "End":
                End(avatar, (x, y), elements)
            x += 32
        y += 32
        x = 0


def blitRotate(surf, image, pos, originpos: tuple, angle: float):
    """
    rotate the player
    :param surf: Surface
    :param image: image to rotate
    :param pos: position of image
    :param originpos: x, y of the origin to rotate about
    :param angle: angle to rotate
    """
    # calcaulate the axis aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]

    # make sure the player does not overlap, uses a few lambda functions(new things that we did not learn about number1)
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])
    # calculate the translation of the pivot
    pivot = Vector2(originpos[0], -originpos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # calculate the upper left origin of the rotated image
    origin = (pos[0] - originpos[0] + min_box[0] - pivot_move[0], pos[1] - originpos[1] - max_box[1] + pivot_move[1])

    # get a rotated image
    rotated_image = pygame.transform.rotozoom(image, angle, 1)

    # rotate and blit the image
    surf.blit(rotated_image, origin)


# def calc_distance(player):
#     global elements
#     min_dist_x = float("inf")
#     obj = None
#     for e in elements:
#         if player.rect.centerx < e.rect.centerx and e.rect.centery < 496 and :
#             cur_dist_x = e.rect.centerx - player.rect.centerx
#             if cur_dist_x < min_dist_x:
#                 min_dist_x = cur_dist_x
#                 obj = e
#     return (min_dist_x, obj)
def calc_distance(player):
    global elements
    min_dist_x = float("inf")
    obj = None
    for e in elements:
        if player.rect.centerx <= e.rect.centerx and e.rect.centery < 496 and player.rect.centery == e.rect.centery:
            cur_dist_x = e.rect.centerx - player.rect.centerx
            cur_dist_y = e.rect.centery - player.rect.centery
            if (cur_dist_x**2 + cur_dist_y**2)**0.5 < min_dist_x:
                min_dist_x = (cur_dist_x**2 + cur_dist_y**2)**0.5
                obj = e
    # print(min_dist_x)
    return (min_dist_x, obj)


def block_map(level_num):
    """
    :type level_num: rect(screen, BLACK, (0, 0, 32, 32))
    open a csv file that contains the right level map
    """
    lvl = []
    with open(level_num, newline='') as csvfile:
        trash = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in trash:
            lvl.append(row)
    return lvl


def reset():
    """resets the sprite groups, music, etc. for death and new level"""
    global player, elements, player_sprite, level, score
    score = 0

    player_sprite = pygame.sprite.Group()
    elements = pygame.sprite.Group()
    # print(avatar)
    player = Player(avatar, elements, (150, 150), player_sprite)

    init_level(
        block_map(
            level_num=levels[level]))


def move_map():
    """moves obstacles along the screen"""
    for sprite in elements:
        sprite.rect.x -= CameraX


def wait_for_key():
    """separate game loop for waiting for a key press while still running game loop
    """
    global level, start
    waiting = True
    while waiting:
        clock.tick(60)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
"""
Global variables
"""
try:
    font = pygame.font.SysFont("lucidaconsole", 20)
except Exception as e:
    print("exception")
# square block face is main character the icon of the window is the block face
try:
    avatar = pygame.image.load(os.path.join("images", "avatar.png"))  # load the main character
    avatar = pygame.transform.smoothscale(avatar, (25, 25))
    alpha_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

    # sprite groups
    player_sprite = pygame.sprite.Group()
    elements = pygame.sprite.Group()

    # images
    spike = pygame.image.load(os.path.join("images", "obj-spike.png"))
    spike = pygame.transform.smoothscale(spike, (32, 32))
    block = pygame.image.load(os.path.join("images", "block_1.png"))
    block = pygame.transform.smoothscale(block, (32, 32))
    orb = pygame.image.load((os.path.join("images", "orb-yellow.png")))
    orb = pygame.transform.smoothscale(orb, (32, 32))
    trick = pygame.image.load((os.path.join("images", "obj-breakable.png")))
    trick = pygame.transform.smoothscale(trick, (32, 32))

except Exception as e:
    print("sdaasds")
pygame.display.set_icon(avatar)
#  this surface has an alpha value with the colors, so the player trail will fade away using opacity

#  ints
fill = 0
num = 0
CameraX = 0
attempts = 0
angle = 0
level = 0

# list
particles = []
orbs = []
win_cubes = []

# Choose level for running
levels = ["level_2.csv", "simple_2_modified.csv", "level_2.csv", "level_2.csv"]

level_list = block_map(levels[level])
level_width = (len(level_list[0]) * 30)
level_height = len(level_list) * 30
init_level(level_list)

# set window title suitable for game
pygame.display.set_caption('Pydash: Geometry Dash in Python')

# initialize the font variable to draw text later
text = font.render('image', False, (255, 255, 0))

# bg image
bg = pygame.image.load(os.path.join("images", "bg.png"))

# create object of player class
# show tip on start and on death
tip = font.render("tip: tap and hold for the first few seconds of the level", True, BLUE)
score = 0
generation = 0


def run(genomes, config):
    reset()
    global keys, angle, score, CameraX, generation
    generation += 1
    # print(generation)
    players = []
    nets = []
    player_sprites = []
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        player_sprites.append(pygame.sprite.Group())
        players.append(Player(avatar, elements, (150, 150), player_sprites[-1]))
    while True:
        score += 1
        for player in players:
            player.vel.x = 6
        alpha_surf.fill((255, 255, 255, 1), special_flags=pygame.BLEND_RGBA_MULT)

        for i, _ in enumerate(players):
            res = calc_distance(players[i])
            output = [0]
            if isinstance(res[1], Spike):
                output = nets[i].activate((players[i].vel.x, players[i].vel.y,
                                           players[i].rect.y, res[0], 1))

            elif isinstance(res[1], Platform):
                output = nets[i].activate((players[i].vel.x, players[i].vel.y,
                                           players[i].rect.y, res[0], 2))
            if output[0] > 0.8:
                if players[i].onGround == True:
                    players[i].isjump = True
                    genomes[i][1].fitness -= 100
                    player_sprites[i].update()

        for i, _ in enumerate(players):
            player_sprites[i].update()
            if players[i].died:
                genomes[i][1].fitness -= 800
                # if genomes[i][1].fitness == -1400 and players[i].win:
                #     genomes[i][1].fitness = 0
                # print(genomes[i][1].fitness)
                players.pop(i)
                player_sprites.pop(i)
                nets.pop(i)
                genomes.pop(i)
        CameraX = 6  # for moving obstacles
        move_map()  # apply CameraX to all elements
        if len(players) == 0:
            break
        screen.blit(bg, (0, 0))  # Clear the screen(with the bg)
        screen.blit(alpha_surf, (0, 0))  # Blit the alpha_surf onto the screen.
        for i, _ in enumerate(players):
            if players[i].isjump:
                """rotate the player by an angle and blit it if player is jumping"""
                angle -= 8.1712  # this may be the angle needed to do a 360 deg turn in the length covered in one jump by player
                blitRotate(screen, players[i].image, players[i].rect.center, (16, 16), angle)
            else:
                """if player.isjump is false, then just blit it normally(by using Group().draw() for sprites"""
                player_sprites[i].draw(screen)  # draw player sprite group
        elements.draw(screen)  # draw all other obstacles

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    # setup config
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    # init NEAT

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    # run NEAT
    p.run(run, 1000)
