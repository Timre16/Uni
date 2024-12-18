import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Labyrinth-Parameter
LABYRINTH_SIZE = (10, 10)  # 10x10-Felder
WALL = -1
FOOD = 1
EMPTY = 0

# Labyrinth erstellen
def generate_labyrinth(size):
    labyrinth = np.zeros(size, dtype=int)
    # Wände zufällig generieren
    for x in range(size[0]):
        for y in range(size[1]):
            if random.random() < 0.2:  # 20% Wahrscheinlichkeit für eine Wand
                labyrinth[x, y] = WALL
    # Essen zufällig platzieren
    for _ in range(5):  # 5 Nahrungsquellen
        x, y = random.randint(0, size[0] - 1), random.randint(0, size[1] - 1)
        labyrinth[x, y] = FOOD
    return labyrinth

# Agent-Umgebung
class SlimeAgent:
    def __init__(self, labyrinth):
        self.labyrinth = labyrinth
        self.position = (random.randint(0, labyrinth.shape[0] - 1), 
                         random.randint(0, labyrinth.shape[1] - 1))
        self.brain = self.create_neural_network()
        self.food_collected = 0

    def create_neural_network(self):
        # Einfaches neuronales Netzwerk mit einem Input für Wände und Essen


        model = Sequential([
            Dense(16, activation="relu", input_shape=(5,)),
            Dense(4, activation="softmax")  # 4 mögliche Bewegungsrichtungen
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def sense_environment(self):
        x, y = self.position
        # Status der vier Wände (oben, unten, links, rechts)
        walls = [
            self.labyrinth[x - 1, y] if x > 0 else WALL,        # Oben
            self.labyrinth[x + 1, y] if x < LABYRINTH_SIZE[0] - 1 else WALL,  # Unten
            self.labyrinth[x, y - 1] if y > 0 else WALL,        # Links
            self.labyrinth[x, y + 1] if y < LABYRINTH_SIZE[1] - 1 else WALL   # Rechts
        ]
        # Ist Nahrung am aktuellen Ort?
        food = 1 if self.labyrinth[x, y] == FOOD else 0
        return np.array(walls + [food])

    def move(self, direction):
        x, y = self.position
        if direction == 0 and x > 0:               # Oben
            self.position = (x - 1, y)
        elif direction == 1 and x < LABYRINTH_SIZE[0] - 1:  # Unten
            self.position = (x + 1, y)
        elif direction == 2 and y > 0:             # Links
            self.position = (x, y - 1)
        elif direction == 3 and y < LABYRINTH_SIZE[1] - 1:  # Rechts
            self.position = (x, y + 1)

    def collect_food(self):
        x, y = self.position
        if self.labyrinth[x, y] == FOOD:
            self.food_collected += 1
            self.labyrinth[x, y] = EMPTY  # Essen konsumiert

    def act(self):
        state = self.sense_environment()
        action_probabilities = self.brain.predict(np.array([state]))[0]
        action = np.argmax(action_probabilities)  # beste Bewegung auswählen
        self.move(action)
        self.collect_food()

# Haupt-Training-Schleife
def train_agent(epochs=1000):
    labyrinth = generate_labyrinth(LABYRINTH_SIZE)
    agent = SlimeAgent(labyrinth)

    for epoch in range(epochs):
        agent.act()
        print(f"Epoch {epoch+1}/{epochs}, Food Collected: {agent.food_collected}")
        if agent.food_collected >= 5:  # alle Nahrung gefunden
            print("Agent hat alle Nahrung gesammelt!")
            break

# Programm ausführen
train_agent()
