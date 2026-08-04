# Passivradar mit FM-Beleuchter und 7-Element-Kreisarray

**Mathematische Grundlagen für die Signalsimulation**
Von der Erzeugung des FM-Signals über die IQ-Daten der 7 Antennenkanäle bis zur
bistatischen Reflexion eines Flugzeugs und der Range-Doppler-Map.

Alle Formeln folgen der Notation von *Malanowski, „Signal Processing for Passive
Bistatic Radar"*; die Gleichungsnummern in Klammern verweisen auf das Buch.

---

## 0. Grundprinzip und Überblick

Ein Passivradar besitzt **keinen eigenen Sender**, sondern nutzt einen fremden
Sender (Illuminator of Opportunity), z. B. einen FM-Rundfunksender. Zur Detektion
werden zwei Signale verglichen:

- **Referenzsignal** $x_r(t)$ — das direkte Signal vom Sender. Es wird über eine
  auf den Sender gerichtete Antenne bzw. einen digital gebildeten Strahl gewonnen.
- **Echosignal** $x_e(t)$ — das vom Ziel (Flugzeug) reflektierte Signal aus dem
  Überwachungsraum.

Die primäre Messgröße ist die **bistatische Entfernung** $R$ (Differenz der Wege
Sender–Ziel–Empfänger und Sender–Empfänger). Zusätzlich wird die
**Doppler-Verschiebung** gemessen, die proportional zur **bistatischen
Geschwindigkeit** $V$ ist.

Die Verarbeitungskette dieser README:

```
FM-Signal erzeugen  →  Modulation / IQ  →  7 UCA-Kanäle (Steering-Vektor)
      →  Bistatische Geometrie (Ziel)  →  Echomodell  →  Cross-Ambiguity
      →  Range-Doppler-Map
```

---

## 1. Der Illuminator of Opportunity: das FM-Signal

FM-Radio arbeitet im Bereich **88–108 MHz** mit hoher Sendeleistung (zehner bis
hunderte kW), was große Reichweite ermöglicht. Der wesentliche Nachteil ist die
**geringe und zeitlich schwankende Bandbreite**: nominell 150 kHz Kanalbreite bei
200 kHz Kanalabstand, effektiv aber je nach Programminhalt zwischen ca. 12 kHz
(Sprache) und ca. 100 kHz (schnelle Musik).

Die zentrale Eigenschaft: Die Information steckt in der **Frequenz**, nicht in der
Amplitude. Der Betrag des Basisbandsignals ist konstant (im Zeitbereich ein
„gesättigter", flacher Betrag).

### 1.1 Signalmodell im Basisband

Das FM-Basisbandsignal ist eine reine Phasenmodulation mit konstantem Betrag 1:

$$x(t) = e^{j\varphi(t)} \tag{2.37}$$

Die **Momentanphase** ist das Integral der Momentanfrequenz:

$$\varphi(t) = 2\pi \int_{-\infty}^{t} f(\tau)\, d\tau \tag{2.38}$$

Umgekehrt ist die **Momentanfrequenz** die abgeleitete Phase geteilt durch $2\pi$:

$$f(t) = \frac{1}{2\pi}\frac{d\varphi(t)}{dt} \tag{2.39}$$

Diese drei Gleichungen sind der Kern der Simulation: Man gibt sich $f(t)$ vor,
integriert zu $\varphi(t)$ und bildet $x(t)=e^{j\varphi(t)}$.

### 1.2 Die echte FM-Momentanfrequenz (Stereo-Multiplex)

Beim realen UKW-Stereo-Signal setzt sich die Momentanfrequenz aus vier
Komponenten zusammen:

$$
f(t) = \Big[\,0{,}9\Big(\tfrac{x_L(t)+x_R(t)}{2}
+ \tfrac{x_L(t)-x_R(t)}{2}\sin(4\pi f_p t)\Big)
+ 0{,}1\sin(2\pi f_p t)
+ \sin(6\pi f_p t)\,x_{rds}(t)\Big]\,\Delta f \tag{2.40}
$$

mit

- $x_L(t), x_R(t)$ — linkes und rechtes Audiokanal-Signal,
- $f_p = 19\,\text{kHz}$ — Pilotfrequenz,
- $\Delta f = 75\,\text{kHz}$ — Frequenzhub.

Die vier Bestandteile:

1. **Mono-Signal** $(x_L+x_R)/2$ — Summensignal (L+R), im Spektrum bei 0–15 kHz.
2. **Stereo-Differenz** $(x_L-x_R)/2\cdot\sin(4\pi f_p t)$ — als DSB-SC
   (Doppelseitenband, unterdrückter Träger) auf 38 kHz moduliert.
3. **Pilotton** $0{,}1\sin(2\pi f_p t)$ — 19 kHz, dient dem Empfänger zur
   Demodulation der Stereo-Differenz.
4. **RDS** $\sin(6\pi f_p t)\cdot x_{rds}(t)$ — digitale Zusatzinfos auf der
   dritten Harmonischen des Pilottons, $3f_p = 57\,\text{kHz}$.

Im demodulierten Spektrum erkennt man daher Peaks bei ±19 kHz (Pilot), das L+R
zwischen ±15 kHz, das L−R um ±38 kHz und RDS um ±57 kHz.

### 1.3 Vereinfachte Modulation für die Simulation

Für eine erste Simulation genügt ein **einzelnes modulierendes Signal** $m(t)$,
normiert auf $[-1,1]$:

$$f(t) = \Delta f \cdot m(t)$$

$m(t)$ ist das „Programm" (Musik/Sprache). Es kann synthetisch sein (Summe einiger
Sinustöne) oder aus echtem Audio stammen. Wichtig ist nur, dass $m(t)$ die
gewünschte Bandbreite besitzt — nicht, ob es wie Musik klingt. Die gesamte
Kette bleibt: $m(t) \to f(t) \to \varphi(t) \to x(t)=e^{j\varphi(t)}$.

### 1.4 Effektive Bandbreite

Weil die FM-Bandbreite inhaltsabhängig schwankt, definiert man eine **effektive
Bandbreite** $B_{ef}$ über die Leistungsdichte $S(f)$ (PSD): Sie ist die Breite
eines Rechtecks, dessen Höhe dem Maximum $S_{max}=\max S(f)$ entspricht und dessen
Fläche gleich der Gesamtleistung ist:

$$S_{max}\cdot B_{ef} = \int_{-\infty}^{+\infty} S(f)\, df \tag{2.41}$$

Diese Größe ist wichtig, weil sie direkt in die Entfernungsauflösung eingeht
(siehe Abschnitt 5.4).

---

## 2. IQ-Daten und was „I" und „Q" bedeuten

Das Signal $x(t)=e^{j\varphi(t)}$ ist **komplexwertig**. Die IQ-Darstellung ist
lediglich die Aufteilung in Real- und Imaginärteil:

$$x(t) = I(t) + j\,Q(t), \qquad I(t)=\cos\varphi(t),\quad Q(t)=\sin\varphi(t)$$

- **I** = In-Phase (Realteil),
- **Q** = Quadrature (Imaginärteil, um 90° phasenverschoben).

Ein IQ-Datenstrom ist also eine Folge komplexer Abtastwerte. Die Abtastrate $f_s$
muss mindestens die Signalbandbreite abdecken (für FM also $\gtrsim 150$ kHz, in
der Praxis z. B. 240 kHz), damit das Multiplex-Spektrum bis 57 kHz sauber
repräsentiert wird (Nyquist).

Bei blockweiser („streaming") Erzeugung ist der wichtigste Punkt, dass die
**Phase $\varphi(t)$ über Blockgrenzen hinweg kontinuierlich** fortgeführt wird.
Startet jeder Block bei Phase 0, entstehen Sprünge, die später die
Cross-Ambiguity-Funktion verfälschen.

---

## 3. Das 7-Element-Kreisarray (UCA)

Bislang war das Signal rein zeitlich. Jetzt kommt die **räumliche** Dimension: Wie
verteilt sich ein aus Richtung $\varphi$ ankommendes Signal auf die 7 Antennen?

### 3.1 Warum ein Kreisarray?

Im Passivradar ist das **uniforme Kreisarray (UCA)** die bevorzugte Anordnung, weil
es durch seine radiale Symmetrie **360°-Rundumsicht** bietet und die Strahlform
nahezu unabhängig vom Steuerwinkel ist. Zum Vergleich: Ein lineares Array (ULA)
hat zwar niedrigere Nebenkeulen (−13 dB gegenüber −8 dB beim UCA) und eine
schmalere Hauptkeule, deckt aber nur einen Sektor (90°–120°) ab, sodass man für
Rundumsicht 3–4 lineare Arrays bräuchte.

### 3.2 Geometrie

Die $N$ Elemente sitzen gleichmäßig auf einem Kreis mit Radius $r$.

- Winkelabstand: $\displaystyle \Delta\varphi = \frac{2\pi}{N}$.
  Für $N=7$: $\Delta\varphi = \tfrac{2\pi}{7} \approx 51{,}43°$.
- Winkel des $n$-ten Elements: $\displaystyle \varphi_n = \frac{2\pi n}{N} = \Delta\varphi \cdot n$, mit $n=0,1,\dots,N-1$.
- Zusammenhang Radius ↔ Elementabstand $d$:

$$r = \frac{d}{2\sin\!\big(\tfrac{\Delta\varphi}{2}\big)} \tag{3.14}$$

  Für $N=7$: $r = \dfrac{d}{2\sin(\pi/7)} \approx 1{,}152\,d$.

### 3.3 Phasenverschiebung und Steering-Vektor

Trifft eine ebene Welle (Fernfeld) aus Azimut $\varphi$ auf das Array, so erfährt
das $n$-te Element gegenüber dem **Array-Zentrum** die Phasenverschiebung:

$$\alpha_n = \frac{2\pi}{\lambda}\, r\,\cos(\varphi_n - \varphi) \tag{3.10}$$

mit Wellenlänge $\lambda = c/f_c$. Der **Steering-Vektor**
$\mathbf{a}(\varphi) = [a_1(\varphi), \dots, a_N(\varphi)]^{\!\top}$ fasst diese
Phasen als komplexe Exponentiale zusammen:

$$a_n(\varphi) = e^{j\alpha_n} = e^{\,j\frac{2\pi}{\lambda} r\cos(\varphi_n - \varphi)} \tag{3.11}$$

Der Steering-Vektor ist das **Herzstück** der Mehrkanal-Simulation: Er ist das
Einzige, was die 7 Kanäle voneinander unterscheidet. Im Fernfeld ist die Amplitude
an allen Elementen gleich; nur die Phasen unterscheiden sich, und zwar exakt nach
(3.11).

### 3.4 IQ-Signal pro Kanal (Direktpfad)

Für **eine** Quelle (z. B. das Direktsignal vom Sender) aus Richtung $\varphi$ mit
Signal $S$ ist das an der Antenne empfangene Vektorsignal:

$$\mathbf{X}(\varphi) = \mathbf{a}(\varphi)\, S \tag{3.20}$$

Für zeitabhängige Signale wird $S \to s(t)$: Jeder Kanal $n$ ist das FM-Signal,
multipliziert mit dem konstanten Phasenfaktor $a_n$:

$$X_n(t) = a_n(\varphi)\cdot s(t)$$

Damit erhältst du aus dem einen FM-Signal $s(t)$ und dem Steering-Vektor die **7
IQ-Datenströme** — eine Matrix der Form $(7 \times \text{Samples})$.

**Sanity-Check:** Die Phasendifferenz zwischen zwei Kanälen ist konstant über die
Zeit und gleich $\alpha_m - \alpha_n$. Das ist der einfachste Test, ob Geometrie
und Signal korrekt zusammenspielen.

### 3.5 Beamforming (spätere Nutzung der 7 Kanäle)

Aus den Kanälen bildet man einen Strahl durch gewichtete Summation:

$$F(\varphi, \mathbf{w}) = \sum_{n=1}^{N} w_n^{*}\, a_n(\varphi) = \mathbf{w}^{H}\mathbf{a}(\varphi) \tag{3.12}$$

wobei $\mathbf{w}=[w_1,\dots,w_N]^{\!\top}$ die komplexen Gewichte sind und $H$ die
konjugiert-transponierte (Hermitesche) Bildung bezeichnet. Um den Strahl in
Richtung $\varphi_0$ zu lenken, kompensiert man die Phasen des Steering-Vektors:

$$w_n = e^{\,j\frac{2\pi}{\lambda} r\cos(\varphi_n - \varphi_0)} \tag{3.13}$$

So werden aus den 7 Kanälen z. B. ein **Referenzstrahl** (auf den Sender gerichtet)
und mehrere **Echostrahlen** (in die Überwachungsrichtungen) gebildet. Der UCA hat
untapert ein Nebenkeulenniveau von ca. −8 dB; mit Tapering (Fensterung der
Gewichte) lässt sich das auf unter −20 dB drücken, allerdings auf Kosten einer
breiteren Hauptkeule und etwas Antennengewinn.

### 3.6 Nichtidealitäten (optional, für Realismus)

In der Realität koppeln die Elemente (mutual coupling) und die Zuleitungen
verzerren Amplitude/Phase. Ein realistischeres Modell:

$$\mathbf{X}_e(\varphi) = \mathbf{C}\,\mathbf{T}\,\mathbf{a}(\varphi)\, S = \hat{\mathbf{C}}\,\mathbf{a}(\varphi)\, S \tag{3.21}$$

mit Kopplungsmatrix $\mathbf{C}$, Zuleitungs-Diagonalmatrix $\mathbf{T}$ und
$\hat{\mathbf{C}}=\mathbf{C}\mathbf{T}$. Ist $\hat{\mathbf{C}}$ bekannt (aus
Kalibrierung), kann das ideale Signal durch Multiplikation mit
$\hat{\mathbf{C}}^{-1}$ zurückgewonnen werden. Für eine erste Simulation setzt man
$\hat{\mathbf{C}}=\mathbf{I}$ (Einheitsmatrix, keine Kopplung).

---

## 4. Bistatische Geometrie: der reflektierende Punkt (Flugzeug)

Jetzt der „nach vorne schauende" Teil: Was passiert, wenn ein Flugzeug das Signal
reflektiert? Zuerst die Geometrie, weil sie festlegt, welche Verzögerung und
welcher Doppler im Echo stecken.

### 4.1 Positionen und Entfernungen

Sei das Ziel bei $(x(t), y(t), z(t))$, der Sender bei $(x_t, y_t, z_t)$, der
Empfänger bei $(x_r, y_r, z_r)$.

- **Sender–Ziel-Entfernung:**
$$R_1(t) = \sqrt{(x-x_t)^2 + (y-y_t)^2 + (z-z_t)^2} \tag{2.1}$$
- **Ziel–Empfänger-Entfernung:**
$$R_2(t) = \sqrt{(x-x_r)^2 + (y-y_r)^2 + (z-z_r)^2} \tag{2.2}$$
- **Basislinie** (Sender–Empfänger):
$$R_b = \sqrt{(x_t-x_r)^2 + (y_t-y_r)^2 + (z_t-z_r)^2} \tag{2.3}$$

Der Winkel zwischen den Segmenten Ziel–Sender und Ziel–Empfänger heißt
**bistatischer Winkel** $\beta$.

### 4.2 Bistatische Entfernung

Die zentrale Messgröße ist die Differenz zwischen indirektem und direktem Weg:

$$R(t) = R_1(t) + R_2(t) - R_b \tag{2.4}$$

Sie ergibt sich aus der gemessenen **Verzögerung** $\tau$ zwischen Echo und Referenz:

$$R = c\,\tau \tag{2.5}$$

Der geometrische Ort konstanter bistatischer Entfernung ist ein **Ellipsoid**
(bzw. in der Ebene eine Ellipse) mit den Brennpunkten Sender und Empfänger.
Ein einzelner $R$-Wert legt das Ziel also nicht eindeutig fest, sondern nur auf
dieser Iso-Range-Ellipse — die zweite Koordinate liefert dann der Einfallswinkel
(aus den 7 Kanälen) oder ein zweites Sender-Empfänger-Paar.

### 4.3 Bistatische Geschwindigkeit und Doppler

Die **bistatische Geschwindigkeit** ist die zeitliche Ableitung der bistatischen
Entfernung:

$$
V(t) = \frac{dR(t)}{dt}
= \frac{(x-x_t)v_x + (y-y_t)v_y + (z-z_t)v_z}{\sqrt{(x-x_t)^2+(y-y_t)^2+(z-z_t)^2}}
+ \frac{(x-x_r)v_x + (y-y_r)v_y + (z-z_r)v_z}{\sqrt{(x-x_r)^2+(y-y_r)^2+(z-z_r)^2}}
\tag{2.6}
$$

Sie hängt also nicht nur von der Position, sondern auch vom
Geschwindigkeitsvektor $(v_x,v_y,v_z)$ des Ziels ab. Der Zusammenhang mit der
gemessenen **Doppler-Verschiebung** $f_d$:

$$V = -\lambda f_d \tag{2.7}$$

Das Minuszeichen: Nimmt die bistatische Entfernung ab ($V<0$), entspricht das einer
positiven Doppler-Frequenz. Bewegt sich das Ziel entlang der Ellipse, ist $V=0$;
senkrecht dazu ist $|V|$ maximal.

---

## 5. Das Echosignal und die Cross-Ambiguity-Funktion

### 5.1 Bewegungsmodell

Die momentane bistatische Entfernung wird als Polynom entwickelt:

$$r(t) = R + Vt + \frac{At^2}{2} + \dots \tag{4.2}$$

mit bistatischer Entfernung $R$, Geschwindigkeit $V$ und Beschleunigung $A$.
Über ein Kohärenzintervall (CPI, Integrationszeit $T$) genügt meist die lineare
Näherung:

$$r(t) \approx R + Vt \tag{4.5}$$

### 5.2 Echomodell

Das an der Antenne empfangene, ins Basisband heruntergemischte Echo eines Ziels
ist eine **verzögerte, Doppler-verschobene und gedämpfte Kopie** des
Referenzsignals:

$$x_e(t) = C''\cdot x_r\!\Big(t - \frac{R}{c}\Big)\cdot \exp\!\Big(j\,\frac{2\pi}{\lambda} V t\Big) \tag{4.7}$$

Dabei ist:

- $x_r(t-R/c)$ — die **Zeitverzögerung** durch den längeren Weg (→ bistatische Entfernung),
- $\exp(j\tfrac{2\pi}{\lambda}Vt)$ — die **Doppler-Verschiebung** durch die Bewegung,
- $C''$ — komplexe Echoamplitude (Dämpfung + Phase).

Für die Simulation heißt das: Das Zielecho entsteht aus demselben FM-Signal
$x_r(t)$, indem man es (a) um $R/c$ verzögert, (b) mit dem Doppler-Faktor
multipliziert, (c) mit dem **Steering-Vektor der Zielrichtung** $\varphi_{tgt}$
auf die 7 Kanäle verteilt (die Zielrichtung ist i. A. **anders** als die
Senderrichtung). Direktsignal und Echo kommen also aus verschiedenen Winkeln und
haben verschiedene Verzögerung/Doppler.

### 5.3 Cross-Ambiguity-Funktion (CAF)

Da $R$ und $V$ des Ziels unbekannt sind, korreliert man das Echosignal mit dem
**verzögerten und Doppler-verschobenen Referenzsignal** über alle vermuteten
$(R,V)$:

$$\psi(R, V) = \int_{-T/2}^{T/2} x_e(t)\cdot x_r^{*}\!\Big(t - \frac{R}{c}\Big)\cdot \exp\!\Big(-j\,\frac{2\pi}{\lambda} V t\Big)\, dt \tag{4.8}$$

mit Integrationszeit (CPI) $T$ und $*$ = komplexe Konjugation.

- Stimmen die probeweise angesetzte Verzögerung und Doppler mit denen eines
  echten Zielechos überein, entsteht ein **Korrelationsgipfel**.
- Berechnet man $\psi(R,V)$ über ein Raster von Entfernungen und Geschwindigkeiten,
  erhält man die **Range-Doppler-Map** (Entfernungs-Geschwindigkeits-Ebene), auf
  der man Ziele als Peaks sucht.

Die CAF entspricht zwei aus dem aktiven Radar bekannten Operationen gleichzeitig:
**Matched Filtering** (erzeugt das Entfernungsprofil) und **Doppler-Processing**
(trennt die Geschwindigkeiten). Da FM kontinuierlich sendet und $T$ in der
Größenordnung 0,1–1 s liegt, ist die Doppler-Auflösung sehr fein und der
Integrationsgewinn groß (kann 50 dB übersteigen).

### 5.4 Auflösungen

- **Bistatische Entfernungsauflösung** — invers zur Bandbreite:
$$\Delta R = c\,\Delta\tau = \frac{c}{B} \tag{2.8}$$
  Für FM mit $B_{ef}\approx 50\,\text{kHz}$ ergibt das $\Delta R \approx 6\,\text{km}$ —
  grob, und wegen der schwankenden FM-Bandbreite auch noch zeitlich variabel.
  **Praxis-Tipp:** Miss die Entfernungsauflösung direkt aus der Breite des
  CAF-Peaks (3-dB-Abfall), statt einen festen Wert anzunehmen.

- **Bistatische Geschwindigkeitsauflösung** — invers zur Integrationszeit:
$$\Delta V = \lambda\,\Delta f_d = \frac{\lambda}{T} \tag{2.9}$$
  Für $\lambda=3\,\text{m}$ (100 MHz) und $T=1\,\text{s}$: $\Delta V \approx 3\,\text{m/s}$ —
  sehr fein. Das ist die Stärke von FM-Passivradar.

### 5.5 Ausblick: warum die reine Map noch nicht reicht (Masking-Effekt)

In einem realen Szenario besteht das Echosignal aus mehreren Komponenten
(Direktsignal, Clutter, Ziele, Rauschen):

$$x_e(t) = \underbrace{\sum_{i=0}^{N_s} C_i^{s}\, x_r\!\Big(t-\tfrac{R_i}{c}\Big)}_{\text{DPI + Clutter (kein Doppler)}} + \underbrace{\sum_{i=1}^{N_m} C_i^{m}\, x_r\!\Big(t-\tfrac{R_i}{c}\Big)\exp\!\Big(j\tfrac{2\pi}{\lambda}V_i t\Big)}_{\text{bewegte Ziele (mit Doppler)}} + \; w(t) \tag{5.1/5.2}$$

Dabei ist die **DPI (Direct-Path Interference)** das direkte Sendesignal (Index
$i=0$, $R_0=0$, kein Doppler) und meist die stärkste Komponente. Das Problem: Die
**zufälligen Nebenkeulen** dieser starken Komponenten liegen bei $BT$ unter dem
Hauptpeak (mit $BT$ = Bandbreite × Integrationszeit, typ. 40–70 dB) und
**überdecken (maskieren) schwache Zielechos**. Ein Flugzeugecho kann so unsichtbar
bleiben, obwohl es über dem Rauschen liegt.

Deshalb folgt in der vollständigen Kette nach der Map (bzw. davor) noch eine
**Direktsignal-/Clutter-Unterdrückung** (adaptive Filter wie NLMS/RLS/LSL oder
Block-Verfahren, alternativ der CLEAN-Algorithmus). Das ist bewusst **nicht** Teil
dieser README — hier endet die Kette bei der Range-Doppler-Map.

---

## 6. Zusammenbau: Simulations-Rezept in Worten

Ohne Code, nur die Reihenfolge der mathematischen Bausteine:

1. **Parameter:** Trägerfrequenz $f_c$ → $\lambda=c/f_c$; Elementzahl $N=7$;
   Radius $r$ (oder Abstand $d$ → (3.14)); Abtastrate $f_s$; Frequenzhub $\Delta f$;
   Integrationszeit $T$.
2. **FM-Signal $s(t)$:** modulierendes $m(t)$ wählen → $f(t)=\Delta f\, m(t)$ →
   $\varphi(t)=2\pi\int f\,dt$ (2.38) → $s(t)=e^{j\varphi(t)}$ (2.37). Phase über
   Blockgrenzen fortführen.
3. **Senderrichtung:** aus Sender- und Empfängerposition den Azimut
   $\varphi_{tx}=\operatorname{atan2}(y_t-y_r,\, x_t-x_r)$ berechnen.
4. **Steering-Vektor Direktpfad:** $\varphi_n=2\pi n/N$; $\alpha_n$ nach (3.10);
   $a_n=e^{j\alpha_n}$ nach (3.11).
5. **7 Direktpfad-Kanäle:** $X_n(t)=a_n(\varphi_{tx})\, s(t)$ (3.20).
6. **Ziel (Flugzeug):** Position + Geschwindigkeit → $R$ (2.4) und $V$ (2.6);
   Zielazimut $\varphi_{tgt}$; Echo nach (4.7) = $s(t)$ verzögert um $R/c$ und
   Doppler-moduliert; auf die 7 Kanäle mit $\mathbf{a}(\varphi_{tgt})$ verteilen.
7. **Summe:** Direktpfad + Echo (+ optional Clutter/Rauschen) pro Kanal addieren.
8. **CAF / Range-Doppler-Map:** Referenz $x_r(t)$ (Direktpfad-Strahl) und Echo
   $x_e(t)$ (Echostrahl) über (4.8) für ein $(R,V)$-Raster korrelieren; Betrag
   $|\psi(R,V)|$ als Map darstellen; Peaks = Ziele.

---

## 7. Welche Mathematik wird benötigt?

Ein kompakter Überblick über das „Werkzeug", das man für die gesamte Kette
beherrschen sollte:

- **Komplexe Zahlen / Zeiger (Phasoren):** Das gesamte Signalmodell lebt in
  $e^{j\varphi}$. IQ = Real-/Imaginärteil. Phasenaddition = Multiplikation von
  Exponentialen. Zentral für FM (2.37), Steering-Vektor (3.11) und Doppler (4.7).
- **Integration / kumulative Summe:** Phase = Integral der Frequenz (2.38); in
  der diskreten Simulation eine kumulative Summe.
- **Ableitung:** Momentanfrequenz (2.39), bistatische Geschwindigkeit als
  Ableitung der Entfernung (2.6).
- **Trigonometrie & analytische Geometrie:** Array-Geometrie (Kosinus in (3.10)),
  Entfernungen und Basislinie (2.1)–(2.3), Azimutberechnung, Ellipsen als
  Iso-Range-Orte.
- **Lineare Algebra:** Steering-Vektor $\mathbf{a}$, Gewichtsvektor $\mathbf{w}$,
  Hermitesches Skalarprodukt $\mathbf{w}^H\mathbf{a}$ (3.12); Matrizen für Kopplung
  (3.21). Die 7 Kanäle sind ein Vektor pro Zeitpunkt.
- **Faltung / Korrelation:** Die CAF (4.8) ist im Kern eine (Doppler-behaftete)
  Kreuzkorrelation von Referenz und Echo = Matched Filter.
- **Fourier-Analyse (FFT):** Spektrum und effektive Bandbreite (2.41); die
  Doppler-Dimension der CAF entsteht durch eine Fouriertransformation über die
  „langsame Zeit". Auch zum Prüfen der simulierten Signale (Multiplex-Peaks).
- **Abtasttheorie (Nyquist):** Wahl von $f_s$ passend zur FM-Bandbreite; Verständnis
  von Range-/Doppler-Auflösung ((2.8), (2.9)) und deren Grenzen.
- **Grundbegriffe der Wahrscheinlichkeit/Statistik** (erst für spätere Stufen mit
  Rauschen und Detektion relevant, hier nicht vertieft).

---

## 8. Symbol- und Formelverzeichnis

| Symbol | Bedeutung | Gl. |
|---|---|---|
| $x(t)=e^{j\varphi(t)}$ | FM-Basisbandsignal (konstanter Betrag) | (2.37) |
| $\varphi(t)$ | Momentanphase = $2\pi\int f\,d\tau$ | (2.38) |
| $f(t)$ | Momentanfrequenz | (2.39), (2.40) |
| $\Delta f$ | Frequenzhub (75 kHz) | (2.40) |
| $f_p$ | Pilotfrequenz (19 kHz) | (2.40) |
| $B_{ef}$ | effektive Bandbreite | (2.41) |
| $\Delta\varphi=2\pi/N$ | Winkelabstand der UCA-Elemente | — |
| $\varphi_n=2\pi n/N$ | Winkel des $n$-ten Elements | — |
| $r$ | Arrayradius | (3.14) |
| $\alpha_n$ | Phasenverschiebung Element $n$ | (3.10) |
| $a_n(\varphi)=e^{j\alpha_n}$ | Steering-Vektor-Element | (3.11) |
| $F(\varphi,\mathbf{w})=\mathbf{w}^H\mathbf{a}$ | Antennendiagramm / Beamforming | (3.12) |
| $w_n$ | Beamforming-Gewicht (Strahl auf $\varphi_0$) | (3.13) |
| $R_1, R_2$ | Sender–Ziel, Ziel–Empfänger | (2.1), (2.2) |
| $R_b$ | Basislinie Sender–Empfänger | (2.3) |
| $R=R_1+R_2-R_b$ | bistatische Entfernung | (2.4) |
| $R=c\tau$ | Entfernung aus Verzögerung | (2.5) |
| $V=dR/dt$ | bistatische Geschwindigkeit | (2.6) |
| $V=-\lambda f_d$ | Geschwindigkeit aus Doppler | (2.7) |
| $\Delta R=c/B$ | Entfernungsauflösung | (2.8) |
| $\Delta V=\lambda/T$ | Geschwindigkeitsauflösung | (2.9) |
| $x_e(t)$ | Echosignal (verzögert, Doppler, gedämpft) | (4.7) |
| $\psi(R,V)$ | Cross-Ambiguity-Funktion | (4.8) |
| $BT$ | Bandbreite × Integrationszeit (Nebenkeulenniveau) | Kap. 5 |

---

*Diese README endet bewusst bei der Range-Doppler-Map. Die nächsten Stufen —
Direktsignal-/Clutter-Unterdrückung, CFAR-Detektion, Richtungsschätzung (DoA) über
die 7 Kanäle und Ziel-Lokalisierung/Tracking — bauen darauf auf und können bei
Bedarf ergänzt werden.*
