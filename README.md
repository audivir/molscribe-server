# molscribe-server

Runs MolScribe as an HTTP API server that predicts a SMILES string or Molfile from an image
of a chemical structure. Includes a command line client, `molscribe-client`, that sends
images to the server and prints or copies the prediction, plus a macOS Automator Quick
Action example for extracting molecules directly from screenshots.

## Prerequisites

- Python 3.10 to 3.14
- PyTorch (a CPU build is sufficient)

## Installation

Install the server:

```shell
pip install torch torchvision torchtext==0.5.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/audivir/molscribe-server
```

Install the client (any Python environment, does not require PyTorch):

```shell
pip install "git+https://github.com/audivir/molscribe-server/#egg=molscribe-client&subdirectory=molscribe-client"
```

## Usage

Run the server:

```shell
python -m molscribe_server
```

Predict from the client:

```shell
molscribe-client smiles <server ip> <server port> image.png
```

Use `molfile` instead of `smiles` to get a Molfile instead. Pass `--use-stdin` to read
input paths from stdin instead of as arguments, and `--use-clipboard` to copy the
prediction to the clipboard instead of printing it.

### macOS Quick Action

Install `terminal-notifier` for feedback notifications:

```shell
brew install terminal-notifier
```

Set up the extraction as a Quick Action:

1. Start `Automator.app`.
1. `File` -> `New` -> `Quick Action`.
1. Configure the workflow to receive `images files` in `any application`.
1. Add a `Run Shell Script` action with `Shell: /bin/zsh` and `Pass input: to stdin`:

```shell
if <python binary> -m molscribe_client smiles <server ip> <server port> --use-stdin --use-clipboard; then
    /opt/homebrew/bin/terminal-notifier -title "Extract SMILES" -message "SMILES extracted successfully."
else
    /opt/homebrew/bin/terminal-notifier -title "Extract SMILES" -message "Failed to extract SMILES." -sound Basso
fi
```

Use `molfile` instead of `smiles` in the script above to extract the Molfile instead.

## License

This project is licensed under the `MIT` license.
