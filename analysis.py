# Bibliotheken, Abhängigkeit Import
import pandas as pd # Pandas für die Verarbeitung von Tabellendaten CSV
import nltk # Hauptbibliothek für Textanalyse
import re # Reguläre Ausdrücke, zum Suchen und Ersetzen von Wörtern
from nltk.corpus import stopwords # Enthält eine listen von Stoppwörtern wie and oder the, die für die Analyse irrelevant sind.
from nltk.stem import WordNetLemmatizer # Wörter werden auf ihre Grundform reduziert, bspw. studies zu study.
from nltk.tokenize import word_tokenize # Token, um ganze Sätze in einzelne Wörter zu zerlegen.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Umwandlung Text in Vektoren 
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation # Themenmodelierung für LSA und LDA

### NLTK Download
# Option quiet=True wird nicht in der Output-Konsole (Terminal) angezeigt. 
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("")
print("Python NLP Analyse - Studenten Beschwerden")
print("Schritt 1. CSV wird gelesen")
df = pd.read_csv("complaints.csv")

"""
In der CSV-Datei sind folgende Spalten enthalten: Genre,Reports,Age,Gpa,Year,Count,Gender,Nationality
Es wird die Spalte „Reports” verwendet und alle leeren Zeilen werden entfernt.
Zudem werden alle Werte in Strings umgewandelt.
Zum Schluss wird die Liste erstellt und die Länge der Liste, also 1005 Datensätze, ausgegeben.
"""
reports = df['Reports'].dropna().astype(str).tolist()
print("Anzahl Datensätze", len(reports))

# Eingebauter Sicherheitscheck stellt sicher, dass die richtigen Daten und Spalten verwendet werden. Er zeigt die erste, zweite und letzte Beschwerde des Datensatzes an.
print("\nAnzeige, Beschwerden 1, 2 und 1005")
print("1. Beschwerde:", reports[0])
print("2. Beschwerde:", reports[1])
print("1005. Beschwerde:", reports[1004])
print("")

### Vorarbeit, Unstrukturierte Texte in saubere Texte umwandeln.
print("Schritt 2. Vorarbeit")

# Lädt Liste von Stoppwörtern wie and oder the, die für die Analyse irrelevant sind.
stop_words = set(stopwords.words('english'))
# Lädt nach, damit sie später ihre Grundform erreichen.
lemmatizer = WordNetLemmatizer()
# Leere Liste
clean = []

"""
In der ersten Schleife wird jeder Text aus der Liste Reports verarbeitet.
Hierbei wird alles kleingeschrieben und alles, was kein Buchstabe ist, also Zahlen, Satz- und Sonderzeichen, entfernt.
Danach wird der Text in einzelne Wörter zerlegt.

In der zweiten Schleife werden Stoppwörter entfernt.Dazu wird zusätzlich ein Filter verwendet, um Wörter wie „a” oder „to” zu entfernen.
Anschließend beginnt die Lemmatisierung, bei der die Wörter in ihre Grundform gebracht werden.
Am Ende werden die Wörter wieder zusammengesetzt und in der Liste clean gespeichert.
"""
for text in reports:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    clean_words = []
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            clean_words.append(lemmatizer.lemmatize(word))
    clean.append(" ".join(clean_words))

### Unstrukturierte Daten in Vektoren umwandeln
print("Schritt 3. Unstrukturierte Daten in Vektoren umwandeln")

# Bag-of-Words-Modell zählt, wie oft ein Wort vorkommt und berücksichtigt die 1000 häufigsten Wörter.
bow_vectorizer = CountVectorizer(max_features=1000)
bow = bow_vectorizer.fit_transform(clean)

# TF-IDF-Modell, kombiniert Häufigkeit und Seltenheit eines Begriffs im Text mit dem Korpus,und berücksichtigt die 1000 häufigsten Wörter.
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(clean)

### Themenextraktion und Ergebnisse in einer Datei speichern
print("Schritt 4. Themenextraktion")

# Die Datei wird geöffnet und das Schreiben beginnt
f = open("ergebnis.txt", "w", encoding="utf-8")
f.write("Python NLP Analyse - Studenten Beschwerden\n")
f.write("Ergebnis\n")
f.write("\n")

"""
Die Funktion top_words hat drei Eingaben matrix, vectorizer und name.
Zunächst wird eine Überschrift erzeugt und in die Datei geschrieben.
Danach werden die Werte aller Wörter summiert, die Wörter abgerufen und mit den Werten verbunden.
Anschließend wird die Wortliste nach Häufigkeit sortiert und die Top 10 angezeigt.
Zum Schluss wird die Ausgabe erzeugt und in der Datei ergebnis.txt gespeichert.
"""
def top_words(matrix, vectorizer, name):
    titel = f"Top 10 Wörter mit {name}"
    f.write(titel + "\n")
    print(f"\n{titel}")
    summen = matrix.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    word_list = list(zip(words, summen))
    word_list.sort(key=lambda x: x[1], reverse=True)
    for wort, value in word_list[:10]:
        line = f"{wort}: {value:.2f}"
        print(line)
        f.write(line + "\n")
    f.write("\n")

"""
Die Funktion topics hat drei Eingabeparameter model, vectorizer und name.
Zunächst wird eine Überschrift erzeugt und in die Datei geschrieben.
Anschließend werden die Wörter geholt und mithilfe einer Schleife werden alle Themen durchgegangen.
Im nächsten Schritt werden die wichtigsten Wörter gefunden, sortiert und die fünf größten Werte gezogen.
Danach werden die Zahlen wieder in Wörter zurückgewandelt.
Am Ende wird alles zu einem Satz zusammengefügt und in der Datei eregebnis.txt gespeichert.
"""
def topics(model, vectorizer, name):
    f.write(f"{name}\n")
    print(f"\n{name}")
    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(model.components_):
        top_words_indices = topic.argsort()[:-6:-1]
        top_words = [words[index] for index in top_words_indices]
        zeile = f"Thema {i+1}: {', '.join(top_words)}"
        print(zeile)
        f.write(zeile + "\n")
    f.write("\n")

### Ausgabe 
print("Schritt 5. Ergebnis ausgeben")

# Es werden die Top-Wörter des Bag-of-Words-Modells und des TF-IDF-Modells angezeigt.
top_words(bow, bow_vectorizer, "Bag-of-Words (Häufige Wörter)")
top_words(tfidf, tfidf_vectorizer, "TF-IDF (Relevante Wörter)")

# Vier Kombinationen des Bag-of-Words-Modells und des TF-IDF-Modells mit den 2 Themenextraktion.
#LSA mit Bag-of-Words, es werden fünf Themen gesucht, die wichtigsten Wörter pro Thema gelernt und anschließend die fünf wichtigsten Wörter pro Thema angezeigt.
lsa_bow = TruncatedSVD(n_components=5, random_state=42)
lsa_bow.fit(bow)
topics(lsa_bow, bow_vectorizer, "LSA mit Bag-of-Words")

# LSA mit TF-IDF, es werden fünf Themen gesucht, die wichtigsten Wörter pro Thema gelernt und anschließend die fünf wichtigsten Wörter pro Thema angezeigt.
lsa_tfidf = TruncatedSVD(n_components=5, random_state=42)
lsa_tfidf.fit(tfidf)
topics(lsa_tfidf, tfidf_vectorizer, "LSA mit TF-IDF")

# LDA mit Bag of Words (Themen aus Wahrscheinlichkeit von Wörtern), es werden fünf Themen gesucht, die wichtigsten Wörter pro Thema gelernt und anschließend die fünf wichtigsten Wörter pro Thema angezeigt.
lda_bow = LatentDirichletAllocation(n_components=5, random_state=42)
lda_bow.fit(bow)
topics(lda_bow, bow_vectorizer, "LDA mit Bag-of-Words")

# LDA mit TF-IDF (Themen aus Wahrscheinlichkeit von Wörtern), es werden fünf Themen gesucht, die wichtigsten Wörter pro Thema gelernt und anschließend die fünf wichtigsten Wörter pro Thema angezeigt.
lda_tfidf = LatentDirichletAllocation(n_components=5, random_state=42)
lda_tfidf.fit(tfidf)
topics(lda_tfidf, tfidf_vectorizer, "LDA mit TF-IDF")

#  Die Datei ergebnis.txt wird geschlossen und gespeichert.
f.close()
print("\nDie Analyse ist abgeschlossen. Die Ergebnisse sind in der Datei ergebnis.txt zu finden.")
print("")