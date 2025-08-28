import csv

# Nom du fichier CSV d'entrée
csv_file = 'Manually_dataset_neurosama.csv'

# Nom du fichier TXT de sortie
txt_file = 'output.txt'

# Ouvrir le fichier CSV en lecture
with open(csv_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    
    # Ouvrir le fichier TXT en écriture
    with open(txt_file, 'w', encoding='utf-8') as txtfile:
        for row in reader:
            for cell in row:
                if cell.strip():  # Vérifie que la cellule n'est pas vide ou composée d'espaces
                    txtfile.write(cell + '\n')

print(f"Le fichier {txt_file} a été créé avec succès, sans lignes vides !")