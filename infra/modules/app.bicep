// Log Analytics Workspace, Container-Apps-Umgebung und die Demo-App.
//
// Referenzen:
//   https://learn.microsoft.com/azure/templates/microsoft.operationalinsights/workspaces?pivots=deployment-language-bicep
//   https://learn.microsoft.com/azure/templates/microsoft.app/managedenvironments?pivots=deployment-language-bicep
//   https://learn.microsoft.com/azure/templates/microsoft.app/containerapps?pivots=deployment-language-bicep

param standort string
param praefix string
param appName string
param image string
param openaiModel string
param minReplicas int

@secure()
param openAiApiKey string

// Der Port stammt aus dem Dockerfile: UVICORN_PORT=8000, EXPOSE 8000.
var zielPort = 8000

resource protokolle 'Microsoft.OperationalInsights/workspaces@2025-07-01' = {
  name: 'log-${praefix}'
  location: standort
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    // 30 Tage ist der kleinste Wert, den die Doku fuer die Voreinstellung eines Workspace
    // nennt. Kuerzer geht nur tabellenweise ueber die API und senkt die Kosten NICHT:
    // 31 Tage Aufbewahrung stecken bereits im Ingestion-Preis.
    // https://learn.microsoft.com/azure/azure-monitor/logs/data-retention-configure
    retentionInDays: 30
  }
}

// Freikontingent der Datenaufnahme: die ersten 5 GB je Abrechnungskonto und Monat kosten
// nichts (Retail-Preis-API, Stufe tierMinimumUnits 5). Ob die Demo darunter bleibt, ist
// OFFEN und wird nach dem ersten Messlauf ueber die Tabelle Usage geprueft -- geschaetzt
// wird hier nichts.

resource umgebung 'Microsoft.App/managedEnvironments@2026-01-01' = {
  name: 'cae-${praefix}'
  location: standort
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: protokolle.properties.customerId
        sharedKey: protokolle.listKeys().primarySharedKey
      }
    }
    // Workload profiles (v2) mit dem Consumption-Profil: laut Doku der Standard und die
    // Empfehlung fuer neue Umgebungen. Regional bestaetigt ueber
    // `az containerapp env workload-profile list-supported -l swedencentral`.
    // https://learn.microsoft.com/azure/container-apps/structure
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

resource anwendung 'Microsoft.App/containerApps@2026-01-01' = {
  name: appName
  location: standort
  properties: {
    managedEnvironmentId: umgebung.id
    workloadProfileName: 'Consumption'
    configuration: {
      activeRevisionsMode: 'Single'
      // Kein registries-Block: Das Demo-Image ist ein oeffentliches GHCR-Paket und zieht
      // ohne Anmeldung. Registry-Zugangsdaten braucht erst ein
      // Interessenten-Image, das privat liegt.
      ingress: {
        external: true
        targetPort: zielPort
        transport: 'auto'
      }
      secrets: [
        {
          name: 'openai-api-key'
          value: openAiApiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'demo'
          // Ausschliesslich per Digest, und zwar auf das PLATTFORM-Manifest, nicht auf den
          // Index. Ein Tag laesst sich verschieben, ein Digest nicht.
          image: image
          // 1,0 vCPU / 2,0 GiB.
          //
          // Der Grund ist der SPEICHER, nicht die CPU: Container Apps koppelt Speicher an
          // CPU im Verhaeltnis 1:2, eine halbe vCPU erzwingt also 1 GiB. Gemessen wurden
          // ein Spitzen-RSS von 0,863 GiB, davon 701 MiB nicht verdraengbar, und in einem
          // Lauf lag der Speicherstand genau auf der 1-GiB-Grenze. Es blieben also rund
          // 14 Prozent Reserve -- in einem Zustand, in dem der Chat-Pfad noch gar nicht
          // gelaufen war. Ein OOM-Kill im ersten Request eines Interessenten ist der
          // schlechteste moegliche Ausgang.
          //
          // Startzeit und Kosten sind Nebenargumente: Bei minReplicas 0 wird nach Aktivzeit
          // bezahlt, und 180.000 vCPU-Sekunden Freikontingent entsprechen bei 1,0 vCPU rund
          // 50 Stunden.
          resources: {
            cpu: json('1.0')
            memory: '2.0Gi'
          }
          env: [
            {
              // Hat in app/config.py keinen Standardwert; ohne ihn scheitert der erste
              // Modellaufruf.
              name: 'OPENAI_MODEL'
              value: openaiModel
            }
            {
              // Ausdruecklich gesetzt, obwohl der Standardwert bereits false ist. Der
              // Schaden, gegen den das steht, ist ein leeres Dashboard ohne Fehlermeldung.
              // Umgestellt wird der Wert in Phase 6/9b, nicht hier.
              name: 'TELEMETRY_ENABLED'
              value: 'false'
            }
            {
              name: 'OPENAI_API_KEY'
              secretRef: 'openai-api-key'
            }
          ]
          // NICHT gesetzt, jeweils weil der Wert in app/config.py schon stimmt. Ein zweiter
          // Ort fuer denselben Wert macht ihn zweideutig statt sicherer:
          //   LLM_PROVIDER      Standardwert openai
          //   LOG_LEVEL         Standardwert INFO
          //   EMBEDDING_DEVICE  Standardwert cpu; ein anderer Wert wird vom Validator abgelehnt
          //   TENANTS_DIR       relativer Standardwert wird gegen die Paketwurzel aufgeloest
          //   INDEX_DIR         und trifft im Image /app/tenants und /app/data/index
          //                     (Begruendung im Dockerfile)
          //   APP_ENV           existiert in Settings, hat aber KEINEN Leser im Code.
          //                     Eine wirkungslose Variable wuerde etwas ueber das System
          //                     behaupten, das nicht stimmt.
          probes: [
            {
              // Startup: ABGELEITET waeren 7 / 1 / 9 -- Budget = 2 x Median(M3), Periode 1 s,
              // initialDelaySeconds = kleinster gemessener M3, failureThreshold = Rest.
              // Letzte Pruefung laege bei 16 s, das Maximum bei Periode 1 und der
              // Spec-Grenze 10 bei 17 s.
              //
              // GESETZT ist bewusst etwas anderes: Periode 5, Schwelle 10, Budget rund 57 s.
              // Eingabe der Ableitung ist ein WARMER lokaler Median; die Cloud ist
              // ungemessen. Liest die Replica das Modell dort kalt, toetet eine zu enge
              // Startup-Probe einen funktionierenden Container.
              //
              // Die Asymmetrie traegt die Wahl: Eine zu enge Startup-Probe bringt ein
              // arbeitendes Replikat um. Eine zu weite verzoegert nur die Erkennung eines
              // defekten -- und sie haelt keinen Verkehr auf, das entscheidet die
              // Readiness-Probe.
              //
              // GEMESSEN am 2026-09-18, und damit entschieden: In der Cloud ist die
              // Anwendung 7 bis 12 s nach dem Containerstart bereit. Das abgeleitete Budget
              // von 17 s haette also gereicht -- aber ohne Reserve. Es bleibt bei 57 s.
              // Einengen braechte nur Risiko: Eine weite Startup-Probe haelt keinen Verkehr
              // auf, das entscheidet die Readiness-Probe.
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: zielPort
              }
              initialDelaySeconds: 7
              periodSeconds: 5
              failureThreshold: 10
              // timeoutSeconds ist eine WAHL, kein Vorgabewert. Vor dem Portbinden scheitert
              // die Probe an der Verbindung, ein Timeout spielt dort keine Rolle. Der Fall
              // "Port bindet, Antwort dauert" ist damit aber nicht abgedeckt: Bei 1 vCPU
              // kann eine Anfrage in Arbeit sein. Gemessen sind rund 20 ms Retrieval je
              // Anfrage und 20 bis 40 ms fuer /health nach dem Binden, ein Timeout ist also
              // unwahrscheinlich -- ausgeschlossen ist er nicht, und unter Dauerlast ist er
              // ungemessen.
              timeoutSeconds: 3
            }
            {
              // Readiness wie abgeleitet. Sie entscheidet, ob Verkehr kommt; ein Fehlschlag
              // startet den Container nicht neu.
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: zielPort
              }
              initialDelaySeconds: 7
              periodSeconds: 5
              failureThreshold: 3
              timeoutSeconds: 3
            }
            {
              // Liveness wie abgeleitet. Sie startet neu, deshalb dieselbe Vorsicht beim
              // Timeout wie oben.
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: zielPort
              }
              initialDelaySeconds: 7
              periodSeconds: 10
              failureThreshold: 3
              timeoutSeconds: 3
            }
          ]
        }
      ]
      scale: {
        // Ein Prozess haelt genau einen Embedder, geteilt ueber Anfragen und Mandanten
        // und er entsteht beim Start des Prozesses. Ein zweites Replikat brachte
        // fuer eine Demo nichts, wuerde aber die Ratenbegrenzung verdoppeln: Der Zaehler in
        // app/ratelimit.py lebt im Prozess.
        //
        // minReplicas kommt als Parameter herein, Standardwert 1 fuer die Akquise-Phase.
        // Gemessen am 2026-09-18: 46 s von der ersten Anfrage bis zur Bereitschaft, wenn die
        // App auf null skaliert war. Das wartet kein Interessent ab.
        //
        // Wer per CLI umstellt und spaeter erneut deployt, landet wieder beim Wert der
        // Vorlage: Scale ist revision-scope, und beim Deployment gewinnt das Template.
        minReplicas: minReplicas
        maxReplicas: 1
      }
    }
  }
}

output umgebungName string = umgebung.name
output appName string = anwendung.name
output fqdn string = anwendung.properties.configuration.ingress.fqdn
output workspaceRessourcenId string = protokolle.id
output workspaceCustomerId string = protokolle.properties.customerId
