.PHONY: protos

# Path to your .proto file(s)
PROTO_DIR = ./protos
PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)

# Output directories
OUT_DIR = HyRLServer/
PYTHON_OUT = $(OUT_DIR)
GRPCLIB_OUT = $(OUT_DIR)
BETTERPROTO_OUT = $(OUT_DIR)

run:
	python server.py

# Command to generate all Python files from .proto
protos:
	uv run python -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(PYTHON_OUT) \
		--grpclib_python_out=$(GRPCLIB_OUT) \
		--python_betterproto_out=$(BETTERPROTO_OUT) \
		$(PROTO_FILES)

# Clean generated files (optional)
clean:
	rm -f $(OUT_DIR)/*.py
