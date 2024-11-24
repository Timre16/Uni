import numpy as np
import matplotlib.pyplot as plt

def generate_binary_values(before_point, after_point):
    binary_values = []
    
    # Generiere Ganzzahlen für Werte >= 1, nur wenn Bits vor dem Komma existieren
    if before_point > 0:
        for i in range(1, 2 ** before_point):  # Start bei 1, damit keine Nachkommastellen entstehen
            binary_values.append(format(i, f'0{before_point}b'))
    
    # Generiere Nachkommazahlen für Werte < 1, nur wenn Bits nach dem Komma existieren
    if after_point > 0:
        after_values = [format(i, f'0{after_point}b') for i in range(1, 2 ** after_point)]
        binary_values.extend([f'0.{a}' for a in after_values])  # Ganzzahldarstellung ist 0
    
    return binary_values

def binary_to_decimal(binary_str):
    # Wandelt binäre Darstellung in Dezimalzahl um
    if '.' in binary_str:
        before, after = binary_str.split('.')
        integer_part = int(before, 2)
        fractional_part = sum(int(bit) * (2 ** -(i + 1)) for i, bit in enumerate(after))
        return integer_part + fractional_part
    else:
        # Nur Ganzzahlteil
        return int(binary_str, 2)

def plot_separate_value_distances(before_point, after_point):
    # Generiere alle möglichen Werte und wandle sie in Dezimalwerte um
    binary_values = generate_binary_values(before_point, after_point)
    decimal_values = sorted([binary_to_decimal(bv) for bv in binary_values])
    
    # Teile Werte in zwei Bereiche auf: < 1 und >= 1
    decimal_values_under_1 = [v for v in decimal_values if v < 1]
    decimal_values_over_1 = [v for v in decimal_values if v >= 1]
    
    # Berechne die Abstände zwischen benachbarten Werten
    distances_under_1 = np.diff(decimal_values_under_1)
    distances_over_1 = np.diff(decimal_values_over_1)
    
    # Plot für Werte < 1
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(decimal_values_under_1[:-1], distances_under_1, color='blue', linestyle='-', marker='o')
    plt.title(f'Abstand zwischen benachbarten Werten für {before_point} Bits vor und {after_point} Bits nach dem Komma (Bereich < 1)')
    plt.xlabel('Dezimalwert')
    plt.ylabel('Abstand zum nächsten Wert')
    plt.grid(True)

    # Plot für Werte ≥ 1
    plt.subplot(2, 1, 2)
    plt.plot(decimal_values_over_1[:-1], distances_over_1, color='green', linestyle='-', marker='o')
    plt.title(f'Abstand zwischen benachbarten Werten für {before_point} Bits vor und {after_point} Bits nach dem Komma (Bereich ≥ 1)')
    plt.xlabel('Dezimalwert')
    plt.ylabel('Abstand zum nächsten Wert')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Eingabe des Bit-Formats und Aufruf der Plotfunktion
input_format = input("Geben Sie das Bit-Format als YYY.XXX ein (z.B., 3.2): ")
before_point, after_point = map(int, input_format.split('.'))
plot_separate_value_distances(before_point, after_point)
