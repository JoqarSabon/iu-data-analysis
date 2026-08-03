# Data Analysis
## Beschreibung

Aufgabenstellung 1. NLP-Techniken anwenden, um eine Textsammlung zu analysieren.
In diesem Projekt sollen unstrukturierte Beschwerdedaten zu einer systematischen Analyse führen. 
Das Ziel besteht darin, am Ende Themen zu extrahieren, um sich um die Beschwerden zu kümmern.

Es wird ein offener Datensatz von Kaggle verwendet, der die Beschwerden simuliert [Students Complaints](https://www.kaggle.com/datasets/omarsobhy14/university-students-complaints-and-reports) (Testdaten).

## Ablauf des Codes

1. CSV wird gelesen
2. Vorarbeit (Unstrukturierte Texte in saubere Texte umwandeln) 
3. Unstrukturierte Daten in Vektoren umwandeln (Bag-of-Words-Modell & TF-IDF-Modell)
4. Themenextraktion (LSA und LDA)
5. Ausgabe, Speicherung in die Datei ergebnis.txt

## Voraussetzungen

- Python 3.12.3
- pandas 3.0.3
- nltk 3.9.4
- scikit-learn 1.9.0

## Projektstruktur

```bash
├── analysis.py # Python Code für die Ausführung
├── complaints.csv # Beschwerde Testdaten
├── ergebnis.txt # Ausgabe aus der Analyse
├── .python-version  # Verwendete Python Version
├── requirements.txt # Python Bibliotheken
├── README.md # Dokumentation
└── .github
    └── workflows
        └── main.yml # GitHub Actions für automatische Ausführung und Security Check

```

##  Installation und Ausführung

```bash
# Vor der Ausführung des Python-Codes sollte die virtuelle Umgebung (venv) aktiviert werden.
# Projekt Klonen
git clone https://github.com/JoqarSabon/iu-data-analysis.git

# In den Ordner gehen
cd iu-data-analysis/

# Bibliotheken installieren
pip install -r requirements.txt

# Ausführen des Codes
python3 analysis.py

# Ergebnis anschauen
cat ergebnis.txt
```

## Github Action

Im Projekt gibt es außerdem eine GitHub-Action-Pipeline, die bei jedem Push im Main-Branch automatisch ausgeführt wird.
Mithilfe der Pipeline wird sichergestellt, dass der Code automatisch ausgeführt, getestet und auf Sicherheitslücken geprüft wird.

Ablauf Pipeline

1. Checkout Repository
2. Einrichtung Python 3.12.3
3. Installation von Bibliotheken
4. Dependency Check mit pip-audit
5. Statische Codeanalyse mit Bandit
6. Ausführung des Code
7. Ergebnisse als GitHub-Artifact hochladen

Nach jeder erfolgreichen Ausführung einer Pipeline werden die Dateien als Artefakt gespeichert und anschließend zum Download bereitgestellt.

- audit.txt
- bandit.txt
- ergebnis.txt
