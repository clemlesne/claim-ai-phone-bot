param adaModel string
param adaVersion string
param gptBackupContext int
param gptBackupModel string
param gptBackupVersion string
param gptContext int
param gptModel string
param gptVersion string
param imageVersion string
param location string
param moderationBlocklists array
param openaiLocation string
param searchLocation string
param tags object

var prefix = deployment().name
var appUrl = 'https://call-center-ai.${acaEnv.properties.defaultDomain}'
var gptBackupModelFullName = toLower('${gptBackupModel}-${gptBackupVersion}')
var gptModelFullName = toLower('${gptModel}-${gptVersion}')
var adaModelFullName = toLower('${adaModel}-${adaVersion}')
var localConfig = loadYamlContent('../config.yaml')
var config = {
  api: {
    events_domain: appUrl
  }
  monitoring: {
    application_insights: {
      connection_string: applicationInsights.properties.ConnectionString
    }
  }
  database: {
    mode: 'cosmos_db'
    cosmos_db: {
      access_key: cosmos.listKeys().primaryMasterKey
      container: container.name
      database: database.name
      endpoint: cosmos.properties.documentEndpoint
    }
  }
  resources: {
    public_url: storageAccount.properties.primaryEndpoints.web
  }
  workflow: {
    default_initiate: {
      bot_company: localConfig.workflow.default_initiate.bot_company
      bot_name: localConfig.workflow.default_initiate.bot_name
      lang: localConfig.workflow.default_initiate.lang
      transfer_phone_number: localConfig.workflow.default_initiate.transfer_phone_number
    }
  }
  communication_services: {
    access_key: communication.listKeys().primaryKey
    endpoint: communication.properties.hostName
    phone_number: localConfig.communication_service.phone_number
  }
  sms: localConfig.sms
  cognitive_service: {
    endpoint: cognitiveCommunication.properties.endpoint
  }
  llm: {
    backup: {
      mode: 'azure_openai'
      azure_openai: {
        context: gptBackupContext
        deployment: gptBackup.name
        endpoint: cognitiveOpenai.properties.endpoint
        model: gptBackupModel
        streaming: true
      }
    }
    primary: {
      mode: 'azure_openai'
      azure_openai: {
        context: gptContext
        deployment: gpt.name
        endpoint: cognitiveOpenai.properties.endpoint
        model: gptModel
        streaming: true
      }
    }
  }
  ai_search: {
    access_key: search.listAdminKeys().primaryKey
    endpoint: 'https://${search.name}.search.windows.net'
    index: 'trainings'
    semantic_configuration: 'default'
  }
  content_safety: {
    access_key: cognitiveContentsafety.listKeys().key1
    blocklists: moderationBlocklists
    endpoint: cognitiveContentsafety.properties.endpoint
  }
  prompts: {
    llm: localConfig.prompts.llm
    tts: localConfig.prompts.tts
  }
  ai_translation: {
    access_key: translate.listKeys().key1
    endpoint: 'https://${translate.name}.cognitiveservices.azure.com/'
  }
  cache: {
    mode: 'redis'
    redis: {
      host: redis.properties.hostName
      password: redis.listKeys().primaryKey
      port: redis.properties.sslPort
    }
  }
}

output appUrl string = appUrl
output blobStoragePublicName string = storageAccount.name
output communicationId string = communication.id
output logAnalyticsWorkspaceCustomerId string = logAnalyticsWorkspace.properties.customerId

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: prefix
  location: location
  tags: tags
  properties: {
    retentionInDays: 30
    sku: {
      name: 'PerGB2018'  // Pay-as-you-go
    }
  }
}

resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: prefix
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

resource acaEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: prefix
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        // Consumption workload profile name must be 'Consumption'
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
  }
}

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'call-center-ai'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8080
      }
    }
    environmentId: acaEnv.id
    template: {
      containers: [
        {
          image: 'ghcr.io/clemlesne/call-center-ai:${imageVersion}'
          name: 'call-center-ai'
          env: [
            {
              name: 'CONFIG_JSON'
              value: string(config)
            }
          ]
          resources: {
            cpu: 1
            memory: '2Gi'
          }
          probes: [
            {
              type: 'Startup'
              tcpSocket: {
                port: 8080
              }
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/health/liveness'
                port: 8080
              }
              periodSeconds: 10  // 2x the timeout
              timeoutSeconds: 5  // Fast to check if the app is running
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health/readiness'
                port: 8080
              }
              periodSeconds: 20  // 2x the timeout
              timeoutSeconds: 10  // Database can take a while to be ready, query is necessary but expensive to run
            }
          ]
        }
      ]
    }
  }
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: replace(prefix, '-', '')
  location: location
  tags: tags
  sku: {
    name: 'Standard_ZRS'
  }
  kind: 'StorageV2'
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: '$web'
}

resource roleCommunicationContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'b24988ac-6180-42a0-ab88-20f7382dd24c'
}

resource appContribCommunication 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, deployment().name, communication.name, 'appContribCommunication')
  scope: communication
  properties: {
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleCommunicationContributor.id
  }
}

resource communication 'Microsoft.Communication/CommunicationServices@2023-06-01-preview' existing = {
  name: prefix
}

resource eventgridTopic 'Microsoft.EventGrid/systemTopics@2023-12-15-preview' = {
  name: prefix
  location: 'global'
  tags: tags
  properties: {
    source: communication.id
    topicType: 'Microsoft.Communication.CommunicationServices'
  }
}

resource roleCognitiveUser 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'a97b65f3-24c7-4388-baec-2e87135dc908'
}

resource appUserCommunication 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, deployment().name, cognitiveCommunication.name, 'appUserCommunication')
  scope: cognitiveCommunication
  properties: {
    principalId: communication.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleCognitiveUser.id
  }
}

resource cognitiveCommunication 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: '${prefix}-${location}-communication'
  location: location
  tags: tags
  sku: {
    name: 'S0'  // Only one available
  }
  kind: 'CognitiveServices'
  properties: {
    customSubDomainName: '${prefix}-${location}-communication'
  }
}

resource cognitiveDocument 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: '${prefix}-${location}-document'
  location: location
  tags: tags
  sku: {
    name: 'S0'  // Pay-as-you-go
  }
  kind: 'FormRecognizer'
  properties: {
    customSubDomainName: '${prefix}-${location}-document'
  }
}

resource cognitiveContentsafety 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: '${prefix}-${location}-contentsafety'
  location: location
  tags: tags
  sku: {
    name: 'S0'  // Pay-as-you-go
  }
  kind: 'ContentSafety'
  properties: {
    customSubDomainName: '${prefix}-${location}-contentsafety'
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

resource roleOpenaiContributor 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  name: 'a001fd3d-188f-4b5d-821b-7da978bf7442'
}

resource appContribOpenai 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(subscription().id, deployment().name, cognitiveOpenai.name, 'appContribOpenai')
  scope: cognitiveOpenai
  properties: {
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: roleOpenaiContributor.id
  }
}

resource cognitiveOpenai 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: '${prefix}-${openaiLocation}-openai'
  location: openaiLocation
  tags: tags
  sku: {
    name: 'S0'  // Pay-as-you-go
  }
  kind: 'OpenAI'
  properties: {
    customSubDomainName: '${prefix}-${openaiLocation}-openai'
  }
}

resource gpt 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: cognitiveOpenai
  name: gptModelFullName
  sku: {
    capacity: 80
    name: 'Standard'  // Pay-as-you-go
  }
  properties: {
    // raiPolicyName: contentfilter.name
    model: {
      format: 'OpenAI'
      name: gptModel
      version: gptVersion
    }
  }
}

resource gptBackup 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: cognitiveOpenai
  name: gptBackupModelFullName
  sku: {
    capacity: 240
    name: 'Standard'  // Pay-as-you-go
  }
  properties: {
    // raiPolicyName: contentfilter.name
    model: {
      format: 'OpenAI'
      name: gptBackupModel
      version: gptBackupVersion
    }
  }
}

// resource contentfilter 'Microsoft.CognitiveServices/accounts/raiPolicies@2023-06-01-preview' = {
//   parent: cognitiveOpenai
//   name: 'disabled'
//   properties: {
//     basePolicyName: 'Microsoft.Default'
//     mode: 'Default'
//     contentFilters: [
//       {
//         blocking: false
//         enabled: false
//         name: 'hate'
//         source: 'Prompt'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'sexual'
//         source: 'Prompt'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'selfharm'
//         source: 'Prompt'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'violence'
//         source: 'Prompt'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'hate'
//         source: 'Completion'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'sexual'
//         source: 'Completion'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'selfharm'
//         source: 'Completion'
//       }
//       {
//         blocking: false
//         enabled: false
//         name: 'violence'
//         source: 'Completion'
//       }
//     ]
//   }
// }

resource ada 'Microsoft.CognitiveServices/accounts/deployments@2023-10-01-preview' = {
  parent: cognitiveOpenai
  name: adaModelFullName
  sku: {
    capacity: 240
    name: 'Standard'  // Pay-as-you-go
  }
  properties: {
    // raiPolicyName: contentfilter.name
    model: {
      format: 'OpenAI'
      name: adaModel
      version: adaVersion
    }
  }
}

resource cosmos 'Microsoft.DocumentDB/databaseAccounts@2023-11-15' = {
  name: prefix
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: {
      defaultConsistencyLevel: 'BoundedStaleness'  // ACID in a single-region, lag in the others
      maxIntervalInSeconds: 600  // 5 mins lags at maximum
      maxStalenessPrefix: 1000  // 1000 requests lags at max
    }
    databaseAccountOfferType: 'Standard'
    locations: [
      {
        locationName: location
      }
    ]
    capabilities: [
      {
        name: 'EnableServerless'
      }
    ]
  }
}

resource database 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-11-15' = {
  parent: cosmos
  name: 'call-center-ai'
  properties: {
    resource: {
      id: 'call-center-ai'
    }
  }
}

resource container 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-11-15' = {
  parent: database
  name: 'calls'
  properties: {
    resource: {
      id: 'calls'
      indexingPolicy: {
        automatic: true
        includedPaths: [
          {
            path: '/created_at/?'
            indexes: [
              {
                dataType: 'String'
                kind: 'Range'
                precision: -1
              }
            ]
          }
          {
            path: '/customer_file/caller_phone/?'
            indexes: [
              {
                dataType: 'String'
                kind: 'Hash'
                precision: -1
              }
            ]
          }
        ]
        excludedPaths: [
          {
            path: '/*'
          }
        ]
      }
      partitionKey: {
        paths: [
          '/initiate/phone_number'
        ]
        kind: 'Hash'
      }
    }
  }
}

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: prefix
  location: searchLocation
  tags: tags
  sku: {
    name: 'basic'  // Smallest with semantic search
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    semanticSearch: 'standard'
  }
}

resource translate 'Microsoft.CognitiveServices/accounts@2023-10-01-preview' = {
  name: '${prefix}-${location}-translate'
  location: location
  tags: tags
  sku: {
    name: 'S1'  // Pay-as-you-go
  }
  kind: 'TextTranslation'
  properties: {
    customSubDomainName: '${prefix}-${location}-translate'
  }
}

resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: prefix
  location: location
  tags: tags
  properties: {
    sku: {
      capacity: 0
      family: 'C'
      name: 'Basic'
    }
  }
}
