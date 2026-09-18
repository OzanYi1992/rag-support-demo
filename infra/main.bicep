// rag-support-demo, Phase 8: Resource Group und Modulaufruf.
//
// Scope ist die Subscription, damit EIN Lauf von `az deployment sub what-if` die Resource
// Group und alles darin zeigt, BEVOR irgendetwas geschrieben wird. Bei zwei getrennten
// Dateien muesste die Gruppe fuer ein group-what-if bereits existieren -- der erste
// Schreibzugriff laege dann vor der ersten Vorschau.
//
// Referenz: https://learn.microsoft.com/azure/azure-resource-manager/bicep/deploy-to-subscription
targetScope = 'subscription'

@description('Region. Entschieden in Phase 8; Begruendung in den Konventionen.')
param standort string = 'swedencentral'

@description('Namenspraefix der subscription-internen Ressourcen.')
param praefix string = 'priv-ragdemo'

@description('Name der Container App. Steht im oeffentlichen FQDN und traegt deshalb kein internes Kuerzel.')
param appName string = 'ca-ragdemo'

@description('Image ausschliesslich per Digest. Kein Tag, kein latest.')
param image string

@description('Modellname des Providers. Ohne ihn scheitert der erste Aufruf, nicht der Start.')
param openaiModel string

@description('Kleinste Replicazahl. 1 fuer die Akquise-Phase; 0 spart die Wochenkosten, kostet aber 46 s Kaltstart.')
param minReplicas int = 1

@secure()
@description('Schluessel des Providers aus LLM_PROVIDER. Wird beim Deployment zugefuehrt und steht in keiner Parameterdatei.')
param openAiApiKey string

resource gruppe 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-${praefix}'
  location: standort
}

module anwendung 'modules/app.bicep' = {
  name: 'ragdemo-anwendung'
  scope: gruppe
  params: {
    standort: standort
    praefix: praefix
    appName: appName
    image: image
    openaiModel: openaiModel
    minReplicas: minReplicas
    openAiApiKey: openAiApiKey
  }
}

output ressourcengruppe string = gruppe.name
output umgebungName string = anwendung.outputs.umgebungName
output appNameAusgabe string = anwendung.outputs.appName
output fqdn string = anwendung.outputs.fqdn
output workspaceRessourcenId string = anwendung.outputs.workspaceRessourcenId
output workspaceCustomerId string = anwendung.outputs.workspaceCustomerId
