# Container configuration
container_name := ghcr.io/clemlesne/call-center-ai
docker := docker
# Versioning
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)
# DevTunnel configuration
tunnel_name := call-center-ai-$(shell hostname | sed 's/[^a-zA-Z0-9]//g' | tr '[:upper:]' '[:lower:]')
tunnel_url ?= $(shell res=$$(devtunnel show $(tunnel_name) | grep -o 'http[s]*://[^"]*' | xargs) && echo $${res%/})
# App location
app_location := westeurope
openai_location := southcentralus
search_location := northeurope
# App configuration
bot_phone_number ?= $(shell cat config.yaml | yq '.communication_services.phone_number')
event_subscription_name ?= $(shell echo '$(name)-$(bot_phone_number)' | tr -dc '[:alnum:]-')
# Bicep outputs
app_url ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["appUrl"].value')
blob_storage_public_name ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["blobStoragePublicName"].value')
communication_id ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["communicationId"].value')
log_analytics_workspace_customer_id ?= $(shell az deployment sub show --name $(name) | yq '.properties.outputs["logAnalyticsWorkspaceName"].value')

version:
	@bash ./cicd/version/version.sh -g . -c

version-full:
	@bash ./cicd/version/version.sh -g . -c -m

install:
	@echo "➡️ Installing Dev Tunnels CLI..."
	devtunnel --version || brew install --cask devtunnel

	@echo "➡️ Installing Allure..."
	allure --version || brew install allure

	@echo "➡️ Installing Python app dependencies..."
	python3 -m pip install -r requirements.txt

	@echo "➡️ Installing Python dev dependencies..."
	python3 -m pip install -r requirements-dev.txt

	@echo "➡️ Testing Docker installation..."
	$(docker) --version || echo "🚨 Docker is not installed."

upgrade:
	@echo "➡️ Upgrading pip..."
	python3 -m pip install --upgrade pip

	@echo "➡️ Upgrading Python app dependencies..."
	python3 -m pur -r requirements.txt

	@echo "➡️ Upgrading Python dev dependencies..."
	python3 -m pur -r requirements-dev.txt

	@echo "➡️ Upgrading Bicep CLI..."
	az bicep upgrade

test:
	@echo "➡️ Running Black..."
	python3 -m black --check .

	@echo "➡️ Running deptry..."
	python3 -m deptry \
		--ignore-notebooks \
		--per-rule-ignores "DEP002=aiohttp|gunicorn|uvicorn" \
		.

	@echo "➡️ Running Pytest..."
	pytest \
		--alluredir test-reports \
		--maxprocesses 4 \
		-n logical \
		-ra \
		tests/*

test-serve:
	allure serve test-reports

lint:
	@echo "➡️ Running Black..."
	python3 -m black .

tunnel:
	@echo "➡️ Creating tunnel..."
	devtunnel show $(tunnel_name) || devtunnel create $(tunnel_name) --allow-anonymous --expiration 7d

	@echo "➡️ Creating port forwarding..."
	devtunnel port show $(tunnel_name) --port-number 8080 || devtunnel port create $(tunnel_name) --port-number 8080

	@echo "➡️ Starting tunnel..."
	devtunnel host $(tunnel_name)

dev:
	VERSION=$(version_full) API__EVENTS_DOMAIN=$(tunnel_url) python3 -m gunicorn main:api \
		--bind 0.0.0.0:8080 \
		--proxy-protocol \
		--reload \
		--reload-extra-file .env \
		--reload-extra-file config.yaml \
		--timeout 120 \
		--worker-class uvicorn.workers.UvicornWorker

build:
	$(docker) build \
		--build-arg VERSION=$(version_full) \
		--tag $(container_name):$(version_small) \
		--tag $(container_name):latest \
		.

start:
	@echo "🛠️ Deploying to localhost..."
	$(docker) run \
		--env API__EVENTS_DOMAIN=$(tunnel_url) \
		--env VERSION=$(version_full) \
		--mount type=bind,source="$(CURDIR)/.env",target="/app/.env" \
		--mount type=bind,source="$(CURDIR)/config.yaml",target="/app/config.yaml" \
		--name call-center-ai \
		--publish 8080:8080 \
		--rm \
		$(container_name):$(version_small)

stop:
	@echo "Stopping container..."
	$(docker) stop call-center-ai

deploy:
	@echo "🛠️ Deploying to Azure..."
	az deployment sub create \
		--location $(app_location) \
		--parameters \
			'openaiLocation=$(openai_location)' \
			'searchLocation=$(search_location)' \
		--template-file bicep/main.bicep \
	 	--name $(name)

	$(MAKE) post-deploy name=$(name)

post-deploy:
	$(MAKE) copy-resources \
		name=$(blob_storage_public_name)

	$(MAKE) eventgrid-register \
		endpoint=$(app_url) \
		name=$(name) \
		source=$(communication_id)

	@echo "🚀 Call Center AI is running on $(app_url)"
	$(MAKE) logs name=$(name)

destroy:
	@echo "🧐 Are you sure you want to delete? Type 'delete now $(name)' to confirm."
	@read -r confirm && [ "$$confirm" = "delete now $(name)" ] || (echo "Confirmation failed. Aborting."; exit 1)

	@echo "❗️ Deleting RG..."
	az group delete --name $(name) --yes --no-wait

	@echo "❗️ Deleting deployment..."
	az deployment sub delete --name $(name)

logs:
	az containerapp logs show \
		--follow \
		--format text \
		--name call-center-ai \
		--resource-group $(name) \
		--tail 100

logs-history:
	az monitor log-analytics query \
		--analytics-query "ContainerAppConsoleLogs_CL | project TimeGenerated, Log_s | sort by TimeGenerated desc" \
		--output jsonc \
		--timespan P1D \
		--workspace $(log_analytics_workspace_customer_id)

eventgrid-register:
	@echo "⚙️ Deleting previous event grid subscription..."
	az eventgrid event-subscription delete --name $(event_subscription_name) || true

	@echo "⚙️ Creating event grid subscription..."
	az eventgrid event-subscription create \
		--advanced-filter data.to.PhoneNumber.Value StringBeginsWith $(bot_phone_number) \
		--enable-advanced-filtering-on-arrays true \
		--endpoint $(endpoint)/call/eventgrid \
		--event-delivery-schema eventgridschema \
		--event-ttl 3 \
		--included-event-types Microsoft.Communication.IncomingCall \
		--max-delivery-attempts 8 \
		--name $(event_subscription_name) \
		--source-resource-id $(source)

copy-resources:
	@echo "📦 Copying resources to Azure storage account..."
	az storage blob upload-batch \
		--account-name $(name) \
		--destination '$$web' \
		--no-progress \
		--output none \
		--overwrite \
		--source resources

watch-call:
	@echo "👀 Watching status of $(phone_number)..."
	while true; do \
		clear; \
		curl -s "$(endpoint)/call?phone_number=%2B$(phone_number)" | yq --prettyPrint '.[0] | {"phone_number": .phone_number, "customer_file": .customer_file, "reminders": .reminders}'; \
		sleep 3; \
	done
