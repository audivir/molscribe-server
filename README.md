# Server

Install the server in a python3.9 environment:
```
pip install torch torchvision torchtext==0.5.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/audivir/molscribe-server
```

Run the server
```
python -m molscribe_server
```

# Client

Install the client in any python environment:
```
pip install "git+https://github.com/audivir/molscribe-server/#egg=molscribe-client&subdirectory=molscribe-client"
```

Install terminal notifier for feedback notifications:
```
brew install terminal-notifier
```
Install the extraction as Quick Action:
1. Start Automator.app
2. File -> New -> Quick Action
3. Configure the workflow:
```
Workflow receives current: `images files` in `any application`
Image: ...
Color: ...
```
4. Add from Utilities -> Run Shell Script: 
```
Shell: `/bin/zsh` Pass input: `to stdin`
```
```
if `python binary` -m molscribe_client smiles `server ip` `server port` --use-stdin --use-clipboard; then
    /opt/homebrew/bin/terminal-notifier -title "Extract SMILES" -message "SMILES extracted successfully."
else
    /opt/homebrew/bin/terminal-notifier -title "Extract SMILES" -message "Failed to extract SMILES." -sound Basso
fi
```

```
if `python binary` -m molscribe_client molfile - `server ip` `server port` --use-stdin --use-clipboard; then
    /opt/homebrew/bin/terminal-notifier -title "Extract Molecule" -message "Molecule extracted successfully."
else
    /opt/homebrew/bin/terminal-notifier -title "Extract Molecule" -message "Failed to extract molecule." -sound Basso
fi
```
