import csv
import math
import os
import random
import sys

import numpy as np
import pygame
from pygame.draw import rect
from pygame.math import Vector2
import matplotlib.pyplot as plt
import time



class GameStop(Exception):
    pass


# The class `NodeGene` represents a node in a neural network.
class NodeGene:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type
        self.value = 0
    def copy(self):
        return NodeGene(self.node_id, self.node_type)


# Class representing a connection between nodes in a neural network
class ConnectionGene:
    def __init__(self, input_node, output_node, weight, enabled=True):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.enabled = enabled



# Class representing a genome in a genetic algorithm for neural networks
class Genome:
    # Constructor method to initialize a Genome object
    def __init__(self, input_nodes, output_nodes):
        self.input_nodes = input_nodes  # List of input nodes in the genome
        self.output_nodes = output_nodes  # List of output nodes in the genome
        self.nodes = input_nodes + output_nodes  # All nodes in the genome
        self.connections = []  # List of connections between nodes
        self.fitness = 0  # Fitness score of the genome
        self.output_bias = 0  # Bias value for output nodes

        # Create connections between input and output nodes with random weights
        for i in range(len(input_nodes)):
            for j in range(len(output_nodes)):
                # Add a new connection gene with random weight to the connections list
                self.connections.append(
                    ConnectionGene(self.input_nodes[i], self.output_nodes[j], weight=np.random.uniform(-10, 10)))

    # Method to activate the neural network with given input values
    def activate_network(self, inputs):
        # Assign input values to input nodes
        for i, input_value in enumerate(inputs):
            self.nodes[i].value = input_value

        # Propagate input values through connections to calculate output values
        for connection in self.connections:
            if connection.enabled:
                connection.output_node.value += connection.weight * connection.input_node.value

        # Apply sigmoid activation function to all output_nodes
        for node in self.output_nodes:
            node.value = self.sigmoid(node.value) + self.output_bias
            if node.value < 0:
                node.value = 0
            elif node.value > 1:
                node.value = 1

        # Calculate output values of output nodes
        output_values = [node.value for node in self.output_nodes]

        return output_values

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def copy(self):
        inp = [i.copy() for i in self.input_nodes]
        out = [i.copy() for i in self.output_nodes]
        copy = Genome(inp, out)
        copy.connections = []
        for i in range(len(inp)):
            for j in range(len(out)):
                # Add a new connection gene with random weight to the connections list
                copy.connections.append(
                    ConnectionGene(inp[i], out[j], weight=self.connections[i].weight))
        return copy


# Class representing the NEAT (NeuroEvolution of Augmenting Topologies)
# algorithm for evolving neural networks
class Neat:
    # Constructor method to initialize a Neat object with the number of inputs, outputs, and population size
    def __init__(self, num_inputs, num_outputs, population_size):
        self.num_inputs = num_inputs  # Number of input nodes in the neural network
        self.num_outputs = num_outputs  # Number of output nodes in the neural network
        self.population_size = population_size  # Size of the population
        self.population = self.create_initial_population()  # Create the initial population of genomes

    # Method to create the initial population of genomes
    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            input_nodes = [NodeGene(i, 'input') for i in range(self.num_inputs)]
            output_nodes = [NodeGene(i + self.num_inputs, 'output') for i in range(self.num_outputs)]
            genome = Genome(input_nodes, output_nodes)
            population.append(genome)
        return population

    # Method to mutate a connection in the genome
    def mutate_connection(self, genome):
        connection = np.random.choice(genome.connections)
        connection.weight += np.random.normal(0, 10)

    # Method to perform crossover between two parent genomes
    def crossover(self, parent1, parent2):
        probs = np.array([parent1.fitness, parent2.fitness])
        probs = probs + np.abs(min(probs)) + 1
        probs = probs / np.sum(probs)

        child_input_nodes = [i.copy() for i in parent1.input_nodes]
        child_output_nodes = [i.copy() for i in parent1.output_nodes]
        child = Genome(child_input_nodes, child_output_nodes)
        child.connections = []
        connections1 = {conn.input_node.node_id: conn for conn in parent1.connections}
        connections2 = {conn.input_node.node_id: conn for conn in parent2.connections}
        for node_id in connections1.keys() | connections2.keys():
            if node_id in connections1 and node_id in connections2:
                connection = np.random.choice([connections1[node_id], connections2[node_id]], p=probs)
                child.connections.append(ConnectionGene(child.nodes[node_id], child.output_nodes[0], weight=connection.weight, enabled=True))
            else:
                raise ValueError()
        return child

    def evolve(self, fitness_function):
        avg_fitness_hist = []
        top_fitness_hist = []
        max_fitness = 0  # Initialize the maximum fitness value
        unchanged_time = 0  # Counter for tracking unchanged fitness
        # top_5_best_performed = []  # List to store the top 5 best-performing genomes
        while max_fitness < 10000:  # Continue evolution until a fitness threshold is reached
            # Evaluate fitness of the population
            try:
                self.evaluate_fitness(fitness_function)
            except GameStop as e:
                return
            probs = np.array(
                [self.population[i].fitness for i in range(self.population_size)])  # Calculate fitness probabilities
            top_fitness_hist.append(np.max(probs))
            avg_fitness_hist.append(np.mean(probs))
            if max_fitness < max(probs):
                max_fitness = max(probs)
                unchanged_time = 0
            if max_fitness < max(probs):
                max_fitness = max(probs)  # Update the maximum fitness value
                unchanged_time = 0
            else:
                unchanged_time += 1
            probs += abs(min(probs)) + 1

            probs = probs / probs.sum()  # Normalize fitness probabilities

            # Select parents and create a new population through crossover and mutation
            parents = [i.copy() for i in self.population]
            new_population = []
            self.population = sorted(self.population, key=lambda x: x.fitness)
            for i in range(-15, 0, 1):
                new_population.append(self.population[i].copy())
            while len(new_population) < self.population_size:
                parent1 = parents[(np.random.choice(range(len(parents)), p=probs))].copy()
                parent2 = parents[(np.random.choice(range(len(parents)), p=probs))].copy()
                if np.random.rand() <= CROSSOVER_RATE:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                if np.random.rand() <= MUTATION_RATE:
                    self.mutate_connection(child)
                new_population.append(child)
            self.population = new_population  # Update the population with the new generation
            if unchanged_time > 200:
                self.population = self.create_initial_population()
        return avg_fitness_hist[:-1], top_fitness_hist[:-1]

    # Method to evaluate the fitness of each genome in the population using the provided fitness function
    def evaluate_fitness(self, fitness_function):
        try:
            self.population = fitness_function(self.population)
        except GameStop as e:
            raise GameStop from e


# Example parameters
NUM_INPUTS = 4
NUM_OUTPUTS = 1
POPULATION_SIZE = 100
GENERATIONS = 10
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.3

# initializes the pygame module
pygame.init()

# creates a screen variable of size 800 x 600
width = 800
height = 600
screen = pygame.display.set_mode([800, 600])

# controls the main game while loop
done = False

# controls whether to start the game from the main menu
start = True

# sets the frame rate of the program
clock = pygame.time.Clock()

# N of generation
generation = 0

# Constant
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

color = lambda: tuple([random.randint(0, 255) for i in range(3)])  # lambda function for random color, not a constant.
GRAVITY = Vector2(0, 0.96)  # Vector2 is a pygame


# Class representing a player sprite in a game
class Player(pygame.sprite.Sprite):
    win: bool  # Flag indicating if the player has won
    died: bool  # Flag indicating if the player has died

    # Constructor method to initialize a Player object with name, image, platforms, position, and groups
    def __init__(self, name, image, platforms, pos, *groups):
        super().__init__(*groups)
        self.onGround = False  # Is the player on the ground?
        self.platforms = platforms  # List of platforms (obstacles)
        self.died = False  # Flag indicating if the player has died
        self.win = False  # Flag indicating if the player has beaten the level

        self.name = name  # Player's name
        self.image = pygame.transform.smoothscale(image, (32, 32))  # Resized player image
        self.rect = self.image.get_rect(center=pos)  # Rectangle representing player's position
        self.jump_amount = 12  # Jump strength
        self.particles = []  # List of particles for player trail
        self.isjump = False  # Is the player jumping?
        self.vel = Vector2(0, 0)  # Player's velocity (starts at zero)
        self.angle = 0  # Player's angle
        self.jumps_count = 0  # Number of jumps made by the player

    # Method to draw a trail of particle-rects behind the player
    def draw_particle_trail(self, x, y, color=(255, 255, 255)):

        # Add a new particle to the trail
        self.particles.append(
            [[x - 5, y - 8], [random.randint(0, 25) / 10 - 1, random.choice([0, 0])],
             random.randint(5, 8)])

        # Update positions of particles and draw them
        for particle in self.particles:
            particle[0][0] += particle[1][0]
            particle[0][1] += particle[1][1]
            particle[2] -= 0.5
            particle[1][0] -= 0.4
            rect(alpha_surf, color, ([int(particle[0][0]), int(particle[0][1])], [int(particle[2]) for i in range(2)]))
            # Remove particles that have faded out
            if particle[2] <= 0:
                self.particles.remove(particle)

    # Method to handle collision detection with obstacles and platforms
    def collide(self, yvel, platforms):
        global coins  # Global variable for coins

        for p in platforms:
            if pygame.sprite.collide_rect(self, p):
                # Check the type of platform
                if isinstance(p, Spike):
                    self.died = True  # Player dies on spike collision

                if isinstance(p, Platform):
                    # Handle collision with normal platforms or half blocks
                    if yvel > 0:
                        # Player is moving downwards
                        self.rect.bottom = p.rect.top  # Prevent player from going through the ground
                        self.vel.y = 0  # Reset vertical velocity as player is on the ground
                        self.onGround = True  # Set onGround flag to true

                        # Reset jump
                        self.isjump = False
                    elif yvel < 0:
                        # Player collides while jumping
                        self.rect.top = p.rect.bottom  # Player's top aligns with bottom of the block
                    else:
                        # Player collides with a block and dies
                        self.vel.x = 0
                        self.rect.right = p.rect.left  # Prevent player from going through walls
                        self.died = True
                if isinstance(p, End):
                    self.win = True  # Player wins when reaching the end

    # Method to make the player jump
    def jump(self):
        self.vel.y = -self.jump_amount  # Set vertical velocity for jumping

    # Method to initiate the jump action
    def doJump(self):
        self.isjump = True  # Set isjump flag to true
        self.jumps_count += 1  # Increment jump count

    # Method to update the player's state
    def update(self):
        """Update the player's state"""

        if self.isjump:
            if self.onGround:
                # Player can only jump if on the ground
                self.jump()

        if not self.onGround:  # Only accelerate with gravity if in the air
            self.vel += GRAVITY  # Apply gravity

            # Limit the maximum falling speed
            if self.vel.y > 100:
                self.vel.y = 100

        # Handle x-axis collisions
        self.collide(0, self.platforms)

        # Update position in the y direction
        self.rect.top += self.vel.y

        # Assume the player is in the air, and set to grounded after collision
        self.onGround = False

        # Handle y-axis collisions
        self.collide(self.vel.y, self.platforms)


# Parent class for all obstacle classes; inherits from the Sprite class
class Draw(pygame.sprite.Sprite):

    # Constructor method to initialize a Draw object with an image, position, and groups
    def __init__(self, image, pos, *groups):
        super().__init__(*groups)
        self.image = image  # Image of the obstacle
        self.rect = self.image.get_rect(topleft=(pos[0], pos[1]))  # Rectangle representing the position of the obstacle
        self.pos = pos  # Position of the obstacle


# Child class representing a platform obstacle
class Platform(Draw):

    # Constructor method to initialize a Platform object with an image, position, and groups
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


# Child class representing a spike obstacle
class Spike(Draw):

    # Constructor method to initialize a Spike object with an image, position, and groups
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


# Child class representing the end point of the level
class End(Draw):

    # Constructor method to initialize an End object with an image, position, and groups
    def __init__(self, image, pos, *groups):
        super().__init__(image, pos, *groups)


# Function to calculate the Euclidean distance between two points a and b
def calc_dist(a, b):
    # Calculate the difference in x and y coordinates
    dx = a[0] - b[0]
    dy = a[1] - b[1]

    # Calculate the Euclidean distance using the Pythagorean theorem
    distance = math.sqrt(dx ** 2 + dy ** 2)

    return distance


spikes = []
spike_groups = []


# Function to initialize the level based on the provided map
def init_level(map):
    x = 0  # Initial x-coordinate
    y = 0  # Initial y-coordinate

    for row in map:
        for col in row:
            if col == "0":
                Platform(block, (x, y), elements)  # Create a Platform object at position (x, y)

            if col == "2":
                s = Spike(spike, [x, y, 1], elements)  # Create a Spike object at position (x, y, 1)

                if spike_groups:
                    last_spike = spikes[-1]
                    if x - last_spike.pos[0] == 32:
                        # Joined spikes (count as one)
                        spike_groups[-1].pos[2] += 1
                    else:
                        spike_groups.append(s)
                else:
                    spike_groups.append(s)

                spikes.append(s)  # Add the Spike object to the list of spikes

            if col == "End":
                End(end, (x, y), elements)  # Create an End object at position (x, y)

            x += 32  # Move to the next column position
        y += 32  # Move to the next row position
        x = 0  # Reset the x-coordinate for the next row

    spike_groups.sort(key=lambda x: x.pos[0])  # Sort the spike groups based on the x-coordinate


# Function to get the position of the closest spike to the player's x-coordinate
def get_closest_spike_position(player_x):
    for s in spike_groups:
        if s.pos[0] > player_x:
            return s.pos
    return None


# Function to get the closest object's distance on the level to the player's position
def get_closest_non_spike_x_coor(travelled_dist, player_y):
    dist = float(15*32)
    for e in elements:
        if e.pos[0] > travelled_dist and e.rect.centery == player_y:
            dist = min(dist, e.pos[0] - travelled_dist)
    return dist


# Function to rotate and draw an image on the surface at a specific position with a given angle
def blitRotate(surf, image, pos, originpos: tuple, angle: float):
    # Calculate the axis-aligned bounding box of the rotated image
    w, h = image.get_size()
    box = [Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
    box_rotate = [p.rotate(angle) for p in box]

    # Calculate the minimum and maximum bounding box points after rotation
    min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
    max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

    # Calculate the translation of the pivot
    pivot = Vector2(originpos[0], -originpos[1])
    pivot_rotate = pivot.rotate(angle)
    pivot_move = pivot_rotate - pivot

    # Calculate the upper-left origin of the rotated image
    origin = (pos[0] - originpos[0] + min_box[0] - pivot_move[0], pos[1] - originpos[1] - max_box[1] + pivot_move[1])

    # Get the rotated image
    rotated_image = pygame.transform.rotozoom(image, angle, 1)

    # Rotate and blit the image on the surface
    surf.blit(rotated_image, origin)


# Function to load and return the map data for a specific level
def block_map(level_num):
    lvl = []
    with open(level_num, newline='') as csvfile:
        trash = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in trash:
            lvl.append(row)
    return lvl


# Function to reset the game state for a new level or player death
def reset():
    global avatars, spikes, spike_groups, passed_distance, elements, level
    elements = pygame.sprite.Group()  # Reset sprite groups
    passed_distance = 150  # Reset travel distance
    spikes = []  # Clear list of spikes
    spike_groups = []  # Clear list of spike groups

    avatars = range(1, 9)  # Reset avatar range

    init_level(block_map(level_num=levels[level]))  # Initialize the level based on the map data


# Function to move obstacles along the screen based on the camera position
def move_map():
    global passed_distance
    passed_distance += CameraX  # Update travel distance based on camera movement

    for sprite in elements:
        sprite.rect.x -= CameraX  # Move each sprite along the x-axis based on camera movement


# Function to resize an image to a specified size
def resize(img, size=(32, 32)):
    resized = pygame.transform.smoothscale(img, size)
    return resized


# Global variables
font = pygame.font.SysFont("lucidaconsole", 20)
dname_font = pygame.font.SysFont("Roboto Condensed", 30)
heading_font = pygame.font.SysFont("Roboto Condensed", 70)

# Load the main character image and set it as the window icon
avatar = pygame.image.load(os.path.join("images", "avatar.png"))
pygame.display.set_icon(avatar)

# Create a surface with alpha for fading player trail
alpha_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

# Initialize sprite groups
elements = pygame.sprite.Group()

# Load and resize images for obstacles
spike = pygame.image.load(os.path.join("images", "obj-spike.png"))
spike = resize(spike)

block = pygame.image.load(os.path.join("images", "block_1.png"))
block = pygame.transform.smoothscale(block, (32, 32))

end = pygame.image.load(os.path.join("images", "finish.png"))
end = pygame.transform.smoothscale(end, (32, 32))

# Initialize variables
fill = 0
num = 0
CameraX = 0
passed_distance = 150
attempts = 0
coins = 0
angle = 0
level = 0

# Lists for particles, orbs, and win cubes
particles = []

# Initialize levels and level data
levels = ["level_to_master.csv", "2.csv", "3.csv"]

# set window title suitable for game
pygame.display.set_caption('Pydash: Geometry Dash in Python')

# initialize the font variable to draw text later
text = font.render('image', False, (255, 255, 0))

# bg image
bg = pygame.image.load(os.path.join("images", "bg.png"))

# avatars
avatars = range(1, 9)
players = []


def run_game(genomes):
    global CameraX, done, players, generation
    dropped_genomes = []
    generation += 1
    players = []
    nets = []
    reset()

    # Initialize genomes
    for g in genomes:
        nets.append(g)
        g.fitness = 0  # Set initial fitness to 0 for each genome
        rand_av = random.choice(avatars)
        sel_avatar = pygame.image.load(os.path.join("images", f"avatars/{rand_av}.png"))
        players.append(Player(rand_av, sel_avatar, elements, (150, 150)))

    # Main game loop
    while not done:
        screen.blit(bg, (0, 0))  # Clear the screen with the background image

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise GameStop

        # Quit if there are no players left
        if len(players) == 0:
            reset()
            break

        # Player controls and neural network activation
        for i, player in enumerate(players):

            genomes[i].fitness += 1.7
            if player.vel.y == 0:
                closest_spike_coords = get_closest_spike_position(passed_distance)
                dist_to_next_obstacle = get_closest_non_spike_x_coor(passed_distance, player.rect.centery)
                if closest_spike_coords is None:
                    output = nets[i].activate_network((
                        40,  # Distance to the nearest spike along the X-axis set to be far
                        0,  # assuming it on the same height
                        1,  # and it is alone spike
                        dist_to_next_obstacle/32
                    ))
                else:
                    output = nets[i].activate_network((
                        (closest_spike_coords[0] - passed_distance)/32,  # Distance to the nearest spike along the X-axis
                        (player.rect.y - closest_spike_coords[1])/32,
                        # Height difference between the player and the nearest spike along the Y-axis
                        closest_spike_coords[2],  # Width of the nearest spike
                        dist_to_next_obstacle/32
                    ))
                if output[0] > 0.5:
                    player.doJump()
                    genomes[i].fitness -= 6.6

        # Update player positions and actions
        for j, player in enumerate(players):
            players[j].vel.x = 6

            player.update()
            CameraX = player.vel.x  # Update CameraX for moving obstacles

            if player.isjump:
                # Rotate the player if jumping
                players[j].angle -= 8.1712  # Angle for a 360-degree turn in the player's jump distance
                blitRotate(screen, player.image, player.rect.center, (16, 16), player.angle)
            else:
                # Blit the player normally if not jumping
                screen.blit(player.image, player.rect)

            if player.died:
                genomes[j].fitness -= 1
                players.pop(j)
                dropped_genomes.append(genomes[j])
                genomes.pop(j)
                nets.pop(j)
            if player.win:
                genomes[j].fitness = 10000
                players.pop(j)
                dropped_genomes.append(genomes[j])
                genomes.pop(j)
                nets.pop(j)

        move_map()  # Move all elements based on CameraX
        elements.draw(screen)  # Draw all other obstacles on the screen

        # Display the current generation on the screen
        label = heading_font.render("Generation: " + str(generation), True, (255, 255, 255))
        label_rect = label.get_rect()
        label_rect.center = (width / 2, 150)
        screen.blit(label, label_rect)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise GameStop
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise GameStop

        pygame.display.flip()  # Update the display
        clock.tick(60)  # Limit the frame rate to 60 FPS

    return dropped_genomes  # Return the dropped genomes for the next generation


if __name__ == "__main__":
    level_str = ""
    instruction_text = font.render("Select index of level to start evolution (from 0)", True, (255, 255, 255))
    level_text = font.render(f"List of levels: {levels}", True, (255, 255, 255))
    input_text = font.render(f"Enter level index: {level_str}", True, (255, 255, 255))

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_BACKSPACE:
                    level_str = level_str[:-1]  # Delete the last character from the input string
                elif event.key == pygame.K_RETURN:
                    # Perform tasks based on the selected level index
                    if level_str:
                        level = int(level_str)
                        if level in range(len(levels)):
                            level_list = block_map(levels[level])
                            level_width = (len(level_list[0]) * 32)
                            level_height = len(level_list) * 32
                            init_level(level_list)
                            print("Starting evolution for level:", level)
                            generation = 0
                            neat_instance = Neat(NUM_INPUTS, NUM_OUTPUTS, POPULATION_SIZE)
                            # Perform tasks for the selected level
                            # Add your code here to start the evolution for the selected level
                            # run NEAT
                            hist = neat_instance.evolve(run_game)
                            iters = range(len(hist[0]))
                            plt.plot(iters, hist[0])
                            plt.title("Average fitness")
                            plt.xlabel("iteration")
                            plt.ylabel("fitness")
                            plt.show()
                            plt.plot(iters, hist[1])
                            plt.title("Highest score")
                            plt.xlabel("iteration")
                            plt.ylabel("fitness")
                            plt.show()
                        else:
                            print(f"Not a valid level index. Please select from {range(len(levels))}")
                    level_str = ""  # Reset the input string after processing the level index
                elif event.key not in [pygame.K_RETURN, pygame.K_BACKSPACE]:
                    level_str += event.unicode  # Append the entered character to the input string

        screen.fill((0, 0, 0))

        # Display the text on the screen
        screen.blit(instruction_text, (50, 50))
        screen.blit(level_text, (50, 100))
        input_text = font.render(f"Enter level index: {level_str}", True, (255, 255, 255))
        screen.blit(input_text, (50, 150))

        # Update the display
        pygame.display.flip()
