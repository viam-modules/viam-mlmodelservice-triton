TAG ?= ghcr.io/USER/REPO:0.0.0

build: Dockerfile module/*
	podman build -t $(TAG) .

push: build
	podman push $(TAG)

run.sh: run.envsubst.sh
	# render entrypoint script from template
	cat $< | TAG=$(TAG) DOLLAR=$$ envsubst > $@
	chmod +x $@

module.tar.gz: run.sh
	# bundle module
	tar czf $@ $^

.PHONY: build push
