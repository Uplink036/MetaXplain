model: ## Run the neural network independantly
	# docker build ./containers/nn/ -t model # uncomment to rebuild
	docker run --network="host" -v ${PWD}/containers/nn:/model model

dataset: ## Load the dataset into the database
	#docker build ./containers/dataset/ -t dataloader # uncomment to rebuild
	docker run --network="host" --env SCRUB_DB=1 -v ${PWD}/containers/dataset:/loader dataloader

compose: ## Run the docker compose
	docker compose up --detach

help: ## Show this help
	@grep -E '^[.a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'