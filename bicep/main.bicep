param adaModel string = 'text-embedding-ada-002'
param adaVersion string = '2'
param agentPhoneNumber string
param botCompany string
param botName string
param gptBackupContext int = 16385
param gptBackupModel string = 'gpt-35-turbo'
param gptBackupVersion string = '0125'
param gptContext int = 128000
param gptModel string = 'gpt-4'
param gptVersion string = '0125-Preview'
param imageVersion string = 'main'
param instance string = deployment().name
param location string = deployment().location
param openaiLocation string
param searchLocation string

targetScope = 'subscription'

output appUrl string = app.outputs.appUrl
output blobStoragePublicName string = app.outputs.blobStoragePublicName
output communicationId string = app.outputs.communicationId
output logAnalyticsWorkspaceCustomerId string = app.outputs.logAnalyticsWorkspaceCustomerId

var tags = {
  application: 'call-center-ai'
  instance: instance
  managed_by: 'Bicep'
  sources: 'https://github.com/clemlesne/call-center-ai'
  version: imageVersion
}

resource sub 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  location: location
  name: instance
  tags: tags
}

module app 'app.bicep' = {
  name: instance
  scope: sub
  params: {
    adaModel: adaModel
    adaVersion: adaVersion
    agentPhoneNumber: agentPhoneNumber
    botCompany: botCompany
    botName: botName
    gptBackupContext: gptBackupContext
    gptBackupModel: gptBackupModel
    gptBackupVersion: gptBackupVersion
    gptContext: gptContext
    gptModel: gptModel
    gptVersion: gptVersion
    imageVersion: imageVersion
    location: location
    moderationBlocklists: []
    openaiLocation: openaiLocation
    searchLocation: searchLocation
    tags: tags
  }
}
