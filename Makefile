dataset:
	docker build ./containers/dataset/ -t dataloader
	docker run --network="host" dataloader 

compose: ## Run the docker compose
	docker compose up

help: ## Show this help
	@grep -E '^[.a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'