HF_REPO := supakornn/mnist-rust

.PHONY: push-github push-hf publish train visualize

train:
	cargo run --release --bin mnist_rust

visualize:
	cargo run --release --bin visualize

push-github:
	git add .
	git commit -m "$(msg)"
	git push

push-hf:
	huggingface-cli upload $(HF_REPO) hf/README.md README.md
	huggingface-cli upload $(HF_REPO) model/ model/
	huggingface-cli upload $(HF_REPO) images/ images/

publish: push-github push-hf
