import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Labyrinth-Parameter
LABYRINTH_SIZE = (10, 10)
WALL = -1
FOOD = 1
EMPTY = 0
AGENT = 2  # Für die visuelle Darstellung des Agenten

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
        self.start_position = (random.randint(0, labyrinth.shape[0] - 1), 
                               random.randint(0, labyrinth.shape[1] - 1))
        self.positions = [self.start_position]  # Alle Positionen des Agenten
        self.food_collected = 0
        self.brain = self.create_neural_network()
        self.reachable_food = set()
        self.collect_food_positions()

    def reset_position(self):
        self.positions = [self.start_position]
        self.food_collected = 0
        self.reachable_food = set()
        self.collect_food_positions()

    def create_neural_network(self):
        # Einfaches neuronales Netzwerk mit einem Input für Wände und Essen
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense

        model = Sequential([
            Dense(16, activation="relu", input_shape=(5,)),
            Dense(4, activation="softmax")  # 4 mögliche Bewegungsrichtungen
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def sense_environment(self, position):
        x, y = position
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

    def expand_one_step(self):
        # Wähle den nächsten erreichbaren Essensort, der noch nicht besucht wurde
        for x, y in self.positions:
            directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Oben, Unten, Links, Rechts
            random.shuffle(directions)  # zufällige Richtung probieren
            for nx, ny in directions:
                if 0 <= nx < LABYRINTH_SIZE[0] and 0 <= ny < LABYRINTH_SIZE[1] and (nx, ny) not in self.positions:
                    if self.labyrinth[nx, ny] != WALL:
                        # Neue Position hinzufügen
                        self.positions.append((nx, ny))
                        if self.labyrinth[nx, ny] == FOOD:
                            self.food_collected += 1
                            self.labyrinth[nx, ny] = EMPTY  # Essen konsumiert
                        return True  # Erfolgreiche Expansion
        return False  # Keine weiteren Expansionsmöglichkeiten
    
    def collect_food_positions(self):
        # Sammle alle Essenspositionen für die Optimierung
        for x in range(LABYRINTH_SIZE[0]):
            for y in range(LABYRINTH_SIZE[1]):
                if self.labyrinth[x, y] == FOOD:
                    self.reachable_food.add((x, y))

    def has_collected_all_food(self):
        # Prüfe, ob alle Essensquellen erreicht wurden
        return self.reachable_food.issubset(self.positions)

    def display_labyrinth(self):
        display_labyrinth = np.copy(self.labyrinth)
        for x, y in self.positions:
            display_labyrinth[x, y] = AGENT
        return display_labyrinth

# Haupt-Trainingsfunktion
def train_agent(epochs=1000):
    labyrinth = generate_labyrinth(LABYRINTH_SIZE)
    agent = SlimeAgent(labyrinth)
    
    fig, ax = plt.subplots()
    ax.set_title("Agent lernt das Labyrinth")

    def update(frame):
        agent.expand_one_step()
        ax.imshow(agent.display_labyrinth(), cmap="viridis")
        ax.set_title(f"Generation: {frame+1} | Food Collected: {agent.food_collected} | Size: {len(agent.positions)}")
        if agent.has_collected_all_food():
            ani.event_source.stop()  # Stoppt die Animation, wenn alle Essensquellen erreicht wurden

    ani = FuncAnimation(fig, update, frames=epochs, repeat=False)
    plt.show()

train_agent(epochs=1000)
