import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np

def bin_to_decimal_fraction(binary_str, weights):
    """Konvertiert eine Binärzahl in eine Dezimalzahl unter Verwendung der angegebenen Gewichte für jedes Bit."""
    decimal_value = 0.0
    for bit, weight in zip(binary_str, weights):
        if bit == '1':
            decimal_value += weight
    return decimal_value

def multiply_custom_binary(bin1, bin2):
    """Multipliziert zwei Binärzahlen mit spezifischen Formaten."""
    # Definiere die Gewichtung für die erste Binärzahl
    weights_bin1 = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]  # Y,XXXXXX

    # Definiere die Gewichtung für die zweite Binärzahl
    weights_bin2 = [8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625]  # 8,4,2,1,0.5,0.25,0.125,0.0625

    # Konvertiere die Binärzahlen in Dezimalwerte
    decimal1 = bin_to_decimal_fraction(bin1, weights_bin1)
    decimal2 = bin_to_decimal_fraction(bin2, weights_bin2)

    # Multipliziere die beiden Werte
    result = decimal1 * decimal2
    return result

# Erzeuge eine Liste aller möglichen 8-Bit-Binärzahlen
all_bin_values = [f"{i:08b}" for i in range(256)]

# Liste für alle Ergebnisse der Multiplikationen
results = []

# Berechne alle möglichen Multiplikationen und speichere die Ergebnisse
for bin1 in all_bin_values:
    for bin2 in all_bin_values:
        result = multiply_custom_binary(bin1, bin2)
        results.append(result)

# Sortiere die Ergebnisse aufsteigend
results_sorted = sorted(results)

# Speichere die Ergebnisse in eine CSV-Datei
output_path = r"C:\Users\timei\Desktop\Python Skripte\Uni\Ausgabe.csv"
with open(output_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Multiplikationsergebnis"])
    for result in results_sorted:
        writer.writerow([result])

# Zeichne die Dichteverteilung der Ergebnisse
plt.figure(figsize=(10, 6))
sns.kdeplot(results, color="blue", fill=True)
plt.xlabel("Multiplikationsergebnis")
plt.ylabel("Dichte")
plt.title("Dichteverteilung der Multiplikationsergebnisse für alle 8-Bit-Binärzahlen")
plt.show()
