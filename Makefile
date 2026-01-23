# Detect operating system
PLATFORM_SHELL := /bin/bash
SCRIPT_EXT := .sh
SCRIPT_DIR := hack

# Borrowed from https://stackoverflow.com/questions/18136918/how-to-get-current-relative-directory-of-your-makefile
curr_dir := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Borrowed from https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
rest_args := $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS))

$(eval $(rest_args):;@:)

# List targets based on script extension and directory
targets := $(shell ls $(curr_dir)/$(SCRIPT_DIR) | grep $(SCRIPT_EXT) | sed 's/$(SCRIPT_EXT)$$//')

$(targets):
	@$(eval TARGET_NAME=$@)
	$(curr_dir)/$(SCRIPT_DIR)/$(TARGET_NAME)$(SCRIPT_EXT) $(rest_args)

help:
	#
	# Usage:
	#
	#   * [dev] `make install`, install development tools, like uv, pre-commit hooks and so on.
	#
	#   * [dev] `make deps`, prepare all dependencies.
	#
	#   * [dev] `make lint`, check style.
	#
	#   * [dev] `make test`, execute unit testing.
	#
	#   * [dev] `make build`, execute building.
	#
	#   * [ci]  `make package`, build container images, not supported on Windows.
	#
	#   * [ci]  `make ci`, execute `make install`, `make deps`, `make lint`, `make test`, `make build`.
	#
	@echo

.DEFAULT_GOAL := build
.PHONY: $(targets)
