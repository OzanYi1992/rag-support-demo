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
// Nachgezogen am 2026-09-24 auf Commit 4120f54. Vorher:
//   sha256:9fa76bc7... (Commit 20351c0, gleicher Tag, wenige Stunden vorher)
//   sha256:25bc3bc7... (Commit bbe8b97, drei Mandanten, 2026-09-23)
//   sha256:591fd63f... (zwei Mandanten, 2026-09-18)
//
// Was dieses Image gegenueber 9fa76bc7 traegt, beides reine Darstellung und
// Route, kein Promptzuwachs:
//   C1  Quelldateien sind klickbar. Neue Route GET /t/{url_token}/doc/{name}.
//       Die Sicherheit steht in ADR-031: Der Mandant kommt aus dem TOKEN, und
//       der Dateiname wird gegen die tatsaechliche Dateiliste aufgeloest, nie zu
//       einem Pfad zusammengesetzt.
//   C2  Der Eskalationskasten nennt die Themen des Mandanten, aus derselben
//       Quelle wie die Begruessung.
//
// Was es ausdruecklich NICHT traegt, unveraendert gegenueber 9fa76bc7: eine
// Knappheitsforderung im Prompt. Sie hat zweimal eine Eigenschaft beschaedigt,
// mit der sie inhaltlich nichts zu tun hat - erst die Antwortsprache, dann das
// Eskalationstor (pitfalls.md, P-030).
//
// Geprueft vor dem Tag, alles mit Gegenprobe:
//   Inhaltsvergleich gegen "git archive 4120f54"  Rueckgabewert 0, 20 Dateien
//   Layer anonym abrufbar, Bereichsabfrage        HTTP 206
//   Attestation vcs:revision                      4120f54555a75b1667...
//   Dokumentroute gegen das Image                 eigenes 200, fremdes 404,
//                                                 jeder Mandant bekommt SEINES
param image = 'ghcr.io/ozanyi1992/rag-support-demo@sha256:e5591132e273d9d67a5b17293a1b5878d76387f985b4483824a5796695725716'

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
