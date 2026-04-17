HF_REPO := supakornn/mnist-rust

.PHONY: push-github push-hf publish train visualize

train:
	cargo run --release --bin mnist_rust

visualize:
	cargo run --release --bin visualize

push-github:
	git add .
	git commit -m "$(if $(msg),$(msg),chore: update)"
	git push

push-hf:
	hf upload $(HF_REPO) hf/README.md README.md
	hf upload $(HF_REPO) model/ model/
	hf upload $(HF_REPO) images/ images/

publish: push-github push-hf
