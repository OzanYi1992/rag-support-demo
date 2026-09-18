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
param image = 'ghcr.io/ozanyi1992/rag-support-demo@sha256:591fd63f83eb7f8e10f526147184fd039c72a888145e7886765b14840743616f'
param openaiModel = 'gpt-5.4-mini-2026-03-17'

// 1 fuer die Akquise-Phase. Gemessen wurden 46 s von der ersten Anfrage bis /health, wenn
// die App auf null skaliert war; das wartet kein Interessent ab. Wer die Outreach-Wellen
// beendet, stellt hier auf 0 zurueck und spart die Wochenkosten.
//
// Der Wert steht hier UND als Standardwert in der Vorlage. Grund: Scale ist
// revision-scope, und ein Deployment mit 0 wuerde eine Umstellung per CLI still
// zuruecksetzen.
param minReplicas = 1

// Gelesen wird die Umgebung des Aufrufs, nicht diese Datei. Fehlt die Variable, bricht die
// Uebersetzung ab -- das ist gewollt: Ein leerer Schluessel wuerde erst im Betrieb auffallen.
param openAiApiKey = readEnvironmentVariable('RAGDEMO_OPENAI_KEY')
