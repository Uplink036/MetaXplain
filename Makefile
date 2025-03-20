meta: ## Run the methamorphical (loader) container
	docker build ./containers/metamorphical/ -t metamorphical # uncomment to rebuild
	docker compose start meta-loader
	
model: ## Run the neural network independantly
	# docker build ./containers/nn/ -t model # uncomment to rebuild
	docker compose start model
dataset: ## Load the dataset into the database
	#docker build ./containers/dataset/ -t dataloader # uncomment to rebuild
	docker compose start loader

database: ## Start only the database
	docker compose start database

compose: ## Run the docker compose
	docker compose up --detach

help: ## Show this help
	@grep -E '^[.a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'