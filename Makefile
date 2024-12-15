.PHONY: e2e_tests_process
e2e_tests_process:
	docker compose -f docker-compose.process.e2e.yaml --profile e2e up --build --exit-code-from e2e-tests-process
