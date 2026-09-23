// Beispielwerte OHNE Geheimnis.
//
// Der Providerschluessel steht NICHT hier und gehoert in keine Datei des Repositories.
// Die Datei nennt nur den NAMEN einer Umgebungsvariablen; den Wert liest die Bicep-CLI
// beim Uebersetzen aus der Umgebung des Aufrufs.
//
// Warum nicht per --parameters inline: Eine .bicepparam-Datei wird fuer sich uebersetzt
// und muss jeden Parameter ohne Standardwert zuweisen (Fehler BCP258). Ein Inline-Wert
// kann eine Zuweisung UEBERSCHREIBEN, aber keine fehlende ergaenzen.
//
// Die Doku zu Parameterdateien warnt zu Recht, dass solche Dateien Werte im Klartext
// halten. Genau deshalb steht hier der Variablenname statt des Schluessels.
using 'main.bicep'

param standort = 'swedencentral'
param praefix = 'priv-ragdemo'
param appName = 'ca-ragdemo'
// Digest, kein Tag. Ein Tag laesst sich verschieben, ein Digest nicht.
//
// Nachgezogen am 2026-09-23 auf das Dreimandantenimage aus Commit bbe8b97.
// Vorher: sha256:591fd63f... (zwei Mandanten, gebaut am 2026-09-18).
//
// Der Inhaltsvergleich gegen "git archive bbe8b97" ist vor dem Tag gelaufen,
// Rueckgabewert 0, mit Gegenprobe. Die Layer sind anonym abrufbar, geprueft
// ueber eine Bereichsabfrage gegen die Registry.
param image = 'ghcr.io/ozanyi1992/rag-support-demo@sha256:25bc3bc724429a45cdfb059b0a0f5a55aa40478e1b94ca3bf0277160ea2e079c'
param openaiModel = 'gpt-5.4-mini-2026-03-17'

// 1 fuer die Akquise-Phase. Gemessen wurden 46 s von der ersten Anfrage bis /health, wenn
// die App auf null skaliert war; das wartet kein Interessent ab. Wer die Outreach-Wellen
// beendet, stellt hier auf 0 zurueck und spart die Wochenkosten.
//
// ENDDATUM 2026-10-31, festgelegt am 2026-09-23. Ein Datum und keine Bedingung: "wenn die
// Wellen beendet sind" ist die Formulierung, bei der nichts passiert. Am 31.10. wird hier
// auf 0 gestellt und neu deployt, unabhaengig davon, wie die Welle gelaufen ist.
//
// Die Zahlen dahinter, gemessen am 2026-09-23: Der Kaltstart betraegt lokal 10,25 s, davon
// 8,2 s Containerstart. In der Cloud kommen 18 bis 21 s Ziehzeit und 12 bis 16 s
// Plattformaktivierung dazu. Scale-to-zero heisst damit knapp eine Minute fuer den ersten
// Klick eines Empfaengers - genau den Klick, auf den die Mail hinauslaeuft.
// Rund fuenf Wochen zu 18,14 USD, also etwa 90 USD fuer die Welle.
//
// Der Wert steht hier UND als Standardwert in der Vorlage. Grund: Scale ist
// revision-scope, und ein Deployment mit 0 wuerde eine Umstellung per CLI still
// zuruecksetzen.
param minReplicas = 1

// Gelesen wird die Umgebung des Aufrufs, nicht diese Datei. Fehlt die Variable, bricht die
// Uebersetzung ab -- das ist gewollt: Ein leerer Schluessel wuerde erst im Betrieb auffallen.
param openAiApiKey = readEnvironmentVariable('RAGDEMO_OPENAI_KEY')
