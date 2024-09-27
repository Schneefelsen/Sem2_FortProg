# Ausarbeitung zur Portfolio-Prüfung in Fortgeschrittene Programmierung:  
# "Sorti" - von Tim Schacht

## Einleitung

Das Programm „Sorti“ ist ein von Supertrumpf inspiriertes Spiel, das den Nutzer in die Welt der Sortier- und Suchalgorithmen eintauchen lässt. Es verbindet unterhaltsame Elemente eines Spiels mit der praktischen Anwendung von Algorithmen, um das Verständnis für deren Effizienz und Anwendung zu fördern. Die Idee hinter „Sorti“ entstand aus der Anforderung der Prüfungsleistung, Sortier- oder Suchalgorithmen miteinander zu vergleichen.
Da Supertrumpf, oder Autoquartett, wie es auch genannt wird, einfach zu verstehen ist, aber gleichzeitig viele Informationen über die für das jeweilige Spiel gewählte Kategorie in einer Rangform miteinander vergleichen kann, eignet sich dieses Spielkonzept gut für das spielerische Erlernen von grundsätzlichen Merkmalen verschiedener Sortier- und Suchalgorithmen.

### Schaffensprozess

Das Ausarbeiten des Programms begann mit einer Recherche über die gängigsten Sortier- und Suchalgorithmen. Ziel war es, Algorithmen auszuwählen, die sowohl in der Theorie als auch in der Praxis von Bedeutung sind. Dabei wurden Algorithmen wie Quicksort, Mergesort, Bubblesort und verschiedene Suchalgorithmen wie binäre Suche und lineare Suche berücksichtigt.
Um die Vielfältigkeit der erdachten Algorithmen hervorzuheben, wurden auch Algorithmen aufgenommen, die besonders für ihre Ineffizienz bekannt sind, wie zum Beispiel Bogo Sort.

Nach der Auswahl der Algorithmen lag der Fokus auf der Entwicklung der Programmarchitektur. Der Entwurf sollte modular und erweiterbar sein, um zukünftige Anpassungen oder Erweiterungen zu erleichtern. Die Struktur des Programms wurde so konzipiert, dass sie eine klare Trennung zwischen den verschiedenen Komponenten gewährleistet, was die Wartung und das Testen der einzelnen Module erleichtert.

## Architektur des Programms

Die Architektur des Programms besteht aus mehreren Schichten, die jede für sich eine spezifische Funktion erfüllen. Die Hauptkomponenten sind die Klassen `StatCard`, die die Informationen über die Algorithmen speichert, und die Klasse `SortiGame`, die das eigentliche Spiel implementiert. 

### Klasse `StatCard`

Die Klasse `StatCard` repräsentiert eine einzelne Karte eines Algorithmus. Jede Karte enthält die Attribute:

- `name`: Der Name des Algorithmus.
- `runtime`: Die Laufzeit des Algorithmus, dargestellt durch eine Punktzahl.
- `complexity`: Die Komplexität des Algorithmus, ebenfalls als Punktzahl dargestellt.
- `usage`: Die Häufigkeit, mit der der Algorithmus verwendet wird.
- `fame`: Die Bekanntheit des Algorithmus.

Die Klasse verfügt über Methoden, um die Statistiken abzurufen und sie in einem ansprechenden Format anzuzeigen. Die visuelle Darstellung der Karten erfolgt durch die `print_card()`-Funktion, das eine klare und strukturierte Ausgabe generiert, um die Merkmale jedes Algorithmus hervorzuheben.

### Klasse `SortiGame`

Die Klasse `SortiGame` ist das Herzstück des Programms und implementiert die Logik des Spiels. Sie steuert den Spielfluss und enthält alle Methoden, die für den Ablauf des Spiels notwendig sind. Die wichtigsten Funktionen der Klasse sind:

- **`get_cards()`**: Diese Methode liest die Statistiken der Algorithmen aus einer CSV-Datei und erstellt für jeden Algorithmus eine `StatCard`. 
- **`choose_fighter()`**: In dieser Methode wählt der Spieler seinen Algorithmus aus den verfügbaren Karten aus. Diese Interaktion kann entweder von einem menschlichen Spieler oder von einem Computer erfolgen. Dabei wird der jeweilige Name der auf den Karten repäsentierten Algorithmen unkenntlich gemacht, wodurch die Spieler auf Basis des Ranges der restlichen Status-Kategorien entscheiden müssen, welcher Algorithmus zu wählen ist.
- **`play_turns()`**: Diese Methode steuert die Durchführung von drei Runden im Spiel. Hierbei wählen die Spieler ihre Algorithmen aus, die dann gegeneinander antreten. Die Leistung der Algorithmen wird anhand ihrer Laufzeiten gemessen.
- **`play()`**: Die Hauptmethode, die das Spiel startet und die Benutzerinteraktionen verwaltet.

Eine besondere Herausforderung stellte die Implementierung der Zeitmessung dar. Hierbei sollte ursprünglich die Multiprocessing-Bibliothek verwendet werden, um die Algorithmen gleichzeitig auszuführen und ihre Laufzeiten zu messen. Dies führte jedoch zu Komplikationen mit jenen Algorithmus-Funktionsdefinitionen, welche selbst Multiprocessing betreiben, weshalb diese Idee wieder verworfen wurde.
Gleichsam stellte sich heraus, dass bei ineffizienten Algorithmen ein forcierter Timeout nötig ist, um das Spiel spielbar zu halten. Diesen zu implementieren hat schlussendlich zur Löschung zweier zuvor implementierter Algorithmen geführt, um Komplikationen mit Multiprocessing zu vermeiden.

### Begleitdateien

Sorti beinhaltet drei essentielle Begleitdateien, namentlich **Search_Algorithms_with_stats.csv**, **Sorting_Algorithms_with_stats.csv** und **util.py**.
Diese stellen den modularen Teil des Programms dar.
In den *CSV*-Dateien befinden sich die tabellarischen Auflistungen der verwendbaren Algorithmen, jeweils versehen mit Angaben von 0-10 zu den Kategorien `name`, `average_runtime`,`implementation_complexity`, `average_worldwide_usage` und `internet_fame`.
In **util.py** werden diese über ein Dictionary mit den dort definierten Funktionen der Such- und Sortieralgorithmen verbunden.


### Benutzeroberfläche

Die Benutzeroberfläche des Spiels ist einfach gehalten und ermöglicht eine intuitive Interaktion. Sie besteht aus Konsolenausgaben, die den Spieler durch das Spiel führen. Der Spieler wird aufgefordert, Entscheidungen zu treffen, während er über die gewählten Algorithmen informiert wird. Dies schafft eine ansprechende und dynamische Spielerfahrung.

Das Spiel bietet die Möglichkeit, gegen einen menschlichen Gegner oder gegen einen Computer anzutreten, was die Wiederspielbarkeit erhöht. Zudem haben die Spieler die Möglichkeit, zwischen Sortier- und Suchalgorithmen zu wählen, was dem Spiel zusätzliche Tiefe verleiht.

## Vergleich der Laufzeiten ausgewählter Algorithmen

Ein zentrales Element des Spiels ist der Vergleich der Laufzeiten der ausgewählten Sortier- und Suchalgorithmen. Dieser Vergleich wird in der Methode `play_turns()` durchgeführt, wo jeder Algorithmus auf eine Liste von 50.000 aufsteigend generierten und im Fall von Sortieralgorithmen anschließend mittels `random.shuffle()` gemischten Zahlen angewendet wird.

Die Laufzeit eines Algorithmus kann erheblich variieren, abhängig von seiner Implementierung und der Art der Daten, die verarbeitet werden. Beispielsweise ist der Quicksort-Algorithmus in der Regel schneller als der Bubblesort-Algorithmus, insbesondere bei großen Datenmengen. Während Quicksort eine durchschnittliche Zeitkomplexität von \(O(n \log n)\) hat, liegt die Zeitkomplexität von Bubblesort bei \(O(n^2)\).

Suchalgorithmen zeigen ähnliche Unterschiede in der Effizienz. Bei einer linearen Suche wird jeder Eintrag nacheinander überprüft, was zu einer Zeitkomplexität von \(O(n)\) führt. Im Gegensatz dazu hat die binäre Suche eine Zeitkomplexität von \(O(\log n)\), benötigt jedoch eine sortierte Liste, um effektiv zu funktionieren. Das Programm demonstriert diese Unterschiede, indem es die Laufzeiten verfolgt und die Spieler dadurch auf die jeweiligen Vorzüge und Nachteile der Algorithmen hinweist.

In diesem Punkt liegt allerdings ebenfalls die zweite große Schwierigkeit der Implementierung dieses Spiels.
Durch die teils stark voneinander abweichenden Einsatzfelder unterschiedlicher Such- und Sortieralgorithmen lassen sich in der endgültigen Form des Programms lediglich solche Algorithmen miteinander vergleichen, welche auf Listen arbeiten.
Dadurch werden unter anderem Algorithmen, die auf binären Suchbäumen arbeiten, nicht betrachtet.
Ebenso musste für einen Vergleich unter selben Startbedingungen bei Suchalgorithmen eine vorsortierte Liste vorgegeben werden, was jedoch denjenigen Suchalgorithmen, die auch chaotische Listen durchsuchen können, zum Nachteil gereicht.


## Fazit

Das Programm „Sorti“ verbindet auf kreative Weise Lernen und Spaß. Durch den Vergleich von Such- und Sortieralgorithmen in einem spielerischen Rahmen wird das Verständnis für algorithmische Effizienz und Datenverarbeitung gefördert. Die modulare Architektur des Programms ermöglicht einfache Erweiterungen, sodass zukünftige Algorithmen und Spielmodi leicht integriert werden können. Insgesamt stellt „Sorti“ eine gute Möglichkeit dar, sich mit der Welt der Algorithmen auseinanderzusetzen und dabei gleichzeitig die Fähigkeiten in der Problemlösung zu verbessern.