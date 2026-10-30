# Running on the PC from the Mac

A Mac cannot use an NVIDIA GPU as a CUDA device, so code and data live on the PC and the Mac
drives it over SSH. Two ways to get the network link; pick one.

## 1. One-time PC setup (Ubuntu 24.04)

```bash
sudo apt update && sudo apt install -y openssh-server tmux git build-essential
sudo ubuntu-drivers install            # NVIDIA driver; reboot afterwards
nvidia-smi                             # should list the GPU
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <this repo> ~/obliterated && cd ~/obliterated && uv sync --extra eval --extra fast
uv run python -m obliterated.device    # device=cuda ...
uv run hf auth login
```

Use password-less SSH: on the Mac run `ssh-keygen -t ed25519` if you have no key, then
`ssh-copy-id <user>@<pc>` once the link below works. Afterwards set `PasswordAuthentication no`
in `/etc/ssh/sshd_config` on the PC.

## 2a. Link with NVIDIA Brev Connect

Brev Connect registers the PC next to any cloud GPUs you rent later, over a NetBird mesh VPN.
It needs systemd, sudo, an SSH server and NVIDIA drivers on the PC (all set up above).

1. In the Brev console, add your own machine and run the install command it gives you on the PC.
2. On the Mac: install the Brev CLI, `brev login`, then `brev ls` should show the PC.
3. `brev shell <pc-name>` opens a terminal; `brev open <pc-name>` opens VS Code over SSH.

The same commands reach a rented H100 for the final large-model check.

## 2b. Link with Tailscale (alternative)

Install Tailscale on both machines and sign in with the same account. Then the PC is reachable
at its tailnet name from anywhere.

## 3. SSH config on the Mac

`~/.ssh/config` (with Brev, `brev` writes an entry for you; use its host name instead):

```
Host obl-pc
    HostName <tailnet-name-or-ip>
    User <user>
    ForwardAgent yes
    ServerAliveInterval 30
    LocalForward 8888 localhost:8888   # Jupyter
```

## 4. Daily loop

- **Editor:** VS Code or Cursor, "Remote-SSH: Connect to Host" → `obl-pc`, open `~/obliterated`.
  Terminals, the Python interpreter and Jupyter kernels all run on the PC's GPU.
- **Long runs:** start them inside tmux so they survive a closed laptop:
  ```bash
  ssh obl-pc
  tmux new -s run
  cd ~/obliterated && uv run python scripts/reproduce_refusal_direction.py --model Qwen/Qwen2.5-7B-Instruct
  # Ctrl-b d to detach; `tmux attach -t run` to come back
  ```
- **Results to the Mac:** pull only summaries and plots, not completions:
  ```bash
  rsync -av --include='*/' --include='summary.json' --include='scores.*' --include='capability_*.json' \
        --include='edit.json' --exclude='*' obl-pc:obliterated/runs/ ./runs/
  ```
- **Tracking:** `uv sync --extra track` and log to Weights & Biases if you want to watch runs from
  the Mac or phone.
- **Git:** commit from either machine; both clones push to the same private repo.
