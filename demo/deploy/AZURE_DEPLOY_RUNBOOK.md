# Runbook: deploy the PRISM-Bio live API to an Azure VM

**Audience:** an AI coding agent running in the owner's local terminal (macOS on Apple Silicon, zsh),
working from a clone of `rishimj/prism-plm`. Follow the phases in order. Every phase ends with a
check; do not continue until it passes. Stop and ask the owner at every **ASK** marker.

**Goal:** run the FastAPI + ESM-2 35M service (`demo/backend`) on the owner's existing Azure
**B2s** VM (2 vCPU, 4 GB RAM), reachable over **HTTPS**, then point the GitHub Pages site at it.

**Ground rules**

1. **The VM is shared with the owner's other projects.** Never stop, remove, restart or reconfigure
   containers, services, proxy configs, firewall rules or DNS labels that this runbook did not
   create. Before any command that touches shared state (ports 80/443, an existing reverse proxy,
   the network security group, the public IP), show the owner what you will change and **ASK**.
2. Build the Docker image **on the VM**, not on the Mac. The Mac is `arm64`, the VM is `x86_64`.
3. Never print, commit or paste secrets (SSH private keys, tokens) into files or chat.
4. Everything this runbook creates is namespaced `prism-bio` (compose project, containers, volumes,
   directory `/opt/prism-plm`, NSG rule names), so it can be found and removed cleanly.
5. If a step fails, diagnose it and report what failed and why. Do not work around a failure by
   loosening security (for example, opening ports to `*` beyond 80/443, or `ALLOWED_ORIGINS=*`).

---

## Phase 0: local prerequisites (Mac)

```bash
# Tools
command -v az   || brew install azure-cli
command -v gh   || brew install gh
command -v jq   || brew install jq
az version --query '"azure-cli"' -o tsv
gh --version | head -1

# Logins (interactive; the owner completes them in the browser)
az account show >/dev/null 2>&1 || az login
gh auth status >/dev/null 2>&1 || gh auth login
```

Locate the repo checkout and make sure the demo code is present:

```bash
cd /path/to/prism-plm            # ASK the owner if you do not know where the clone is
git fetch origin
git branch -a | grep -E 'main|claude/biology-demo-recruiter-52550g'
ls demo/backend/Dockerfile demo/deploy/docker-compose.yml demo/backend/assets/concepts.json
```

**Which branch to deploy.** The demo was developed on `claude/biology-demo-recruiter-52550g`. If it
has been merged, use `main`; otherwise use the feature branch.

```bash
if git merge-base --is-ancestor origin/claude/biology-demo-recruiter-52550g origin/main 2>/dev/null; then
  export DEPLOY_BRANCH=main
else
  export DEPLOY_BRANCH=claude/biology-demo-recruiter-52550g
fi
echo "Deploying branch: $DEPLOY_BRANCH"
```

**Check:** `az account show` and `gh auth status` succeed, and the three files listed above exist.

---

## Phase 1: identify the VM

```bash
az vm list -d -o table --query "[].{name:name, rg:resourceGroup, size:hardwareProfile.vmSize, state:powerState, ip:publicIps, fqdn:fqdns, os:storageProfile.osDisk.osType}"
```

- If exactly one VM has size `Standard_B2s` (or similar), propose it. Otherwise **ASK** which VM to use.
- The VM must be running (`VM running`). If it is deallocated, **ASK** before starting it
  (`az vm start -g "$RG" -n "$VM"`).

```bash
export RG=<resource-group>
export VM=<vm-name>

# Details we need later
export ADMIN_USER=$(az vm show -g "$RG" -n "$VM" --query osProfile.adminUsername -o tsv)
NIC_ID=$(az vm show -g "$RG" -n "$VM" --query 'networkProfile.networkInterfaces[0].id' -o tsv)
export PIP_ID=$(az network nic show --ids "$NIC_ID" --query 'ipConfigurations[0].publicIPAddress.id' -o tsv)
export PUBLIC_IP=$(az network public-ip show --ids "$PIP_ID" --query ipAddress -o tsv)
export EXISTING_FQDN=$(az network public-ip show --ids "$PIP_ID" --query dnsSettings.fqdn -o tsv)
export NIC_NSG=$(az network nic show --ids "$NIC_ID" --query networkSecurityGroup.id -o tsv)
SUBNET_ID=$(az network nic show --ids "$NIC_ID" --query 'ipConfigurations[0].subnet.id' -o tsv)
export SUBNET_NSG=$(az network vnet subnet show --ids "$SUBNET_ID" --query networkSecurityGroup.id -o tsv)
echo "user=$ADMIN_USER ip=$PUBLIC_IP fqdn=${EXISTING_FQDN:-<none>}"
echo "nic nsg=${NIC_NSG:-<none>}"; echo "subnet nsg=${SUBNET_NSG:-<none>}"
```

**Check:** `PUBLIC_IP` is non-empty. If the VM has no public IP, stop and **ASK** (this runbook
assumes a public IP).

---

## Phase 2: SSH access and a survey of the VM (read-only)

```bash
ssh -o StrictHostKeyChecking=accept-new "$ADMIN_USER@$PUBLIC_IP" 'echo ok; uname -m; lsb_release -ds 2>/dev/null || cat /etc/os-release | head -2'
```

If SSH fails, try `az ssh vm -g "$RG" -n "$VM"` or **ASK** the owner which key or host alias they
use (check `~/.ssh/config` for a host entry pointing at `$PUBLIC_IP`). Once it works, set:

```bash
export SSH_TARGET="$ADMIN_USER@$PUBLIC_IP"     # or the ~/.ssh/config alias
```

Survey what is already running. **Do not change anything in this phase.**

```bash
ssh "$SSH_TARGET" 'bash -s' <<'EOF'
echo "== arch / memory / disk"; uname -m; free -h; df -h / | tail -1
echo "== swap"; swapon --show
echo "== docker"; (command -v docker && docker --version && docker compose version) || echo "docker: not installed"
echo "== containers"; sudo docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}' 2>/dev/null || true
echo "== listeners on 80/443/8000"; sudo ss -tlnp | awk 'NR==1 || /:(80|443|8000) /'
echo "== web servers"; for s in nginx caddy apache2 traefik; do systemctl is-active --quiet $s && echo "$s: active (systemd)"; done
echo "== existing prism deploy"; ls -d /opt/prism-plm 2>/dev/null || echo none
EOF
```

Decide the **deployment mode** from the survey:

| What is listening on 80/443 | Mode |
|---|---|
| Nothing | **Mode A**: the bundled Caddy container owns 80/443 and gets the TLS certificate |
| An existing nginx / Caddy / Traefik (host service or container) | **Mode B**: run only the API on `127.0.0.1:8000` and add one route to the existing proxy |
| Something unclear | **ASK** |

Also check port 8000: if something already listens on it, pick another port for Mode B and export
`API_PORT=<free port>` (used by `docker-compose.behind-proxy.yml`).

**Memory:** the API uses about 1 GB. If `free -h` shows less than ~1.5 GB available, tell the owner
before continuing (the setup adds a 2 GB swap file, which helps with the image build but not with
steady-state memory pressure).

Report the survey and the chosen mode to the owner before Phase 3.

---

## Phase 3: choose the public hostname

The site is served over HTTPS from GitHub Pages, so browsers will only call an **HTTPS** API with a
valid certificate. The certificate needs a hostname that resolves to `$PUBLIC_IP`.

Pick the first option that applies:

1. **The owner has a custom domain** they want to use (for example `api.example.com`). **ASK**, then
   have them create an `A` record to `$PUBLIC_IP`. Use it as `API_DOMAIN`.
2. **The public IP has no DNS label** (`EXISTING_FQDN` is empty). **ASK** before adding one, then:
   ```bash
   az network public-ip update --ids "$PIP_ID" --dns-name prism-bio-$RANDOM
   export API_DOMAIN=$(az network public-ip show --ids "$PIP_ID" --query dnsSettings.fqdn -o tsv)
   ```
3. **The IP already has a DNS label used by other projects.** Do not change it. Use a dedicated
   wildcard-DNS hostname that maps to the same IP, so the API gets its own certificate and cannot
   collide with other sites:
   ```bash
   export API_DOMAIN="prism-bio.$(echo $PUBLIC_IP | tr . -).sslip.io"
   ```

**Note:** if the public IP is *dynamic*, it can change when the VM is deallocated. Options 2 and 3
depend on the IP. Check with
`az network public-ip show --ids "$PIP_ID" --query publicIPAllocationMethod -o tsv`; if it is
`Dynamic`, tell the owner (making it static may briefly interrupt other projects, so **ASK**).

```bash
echo "API_DOMAIN=$API_DOMAIN"
dig +short "$API_DOMAIN"        # must print $PUBLIC_IP
```

**Check:** `dig` returns `$PUBLIC_IP`.

---

## Phase 4: open inbound ports 80 and 443 (Azure network security group)

Check whether the NSG(s) already allow them. There may be an NSG on the NIC, the subnet, or both.
Traffic must be allowed by **every** NSG in the path.

```bash
for NSG in $NIC_NSG $SUBNET_NSG; do
  echo "== $NSG"
  az network nsg rule list --ids "$NSG" -o table \
    --query "[?direction=='Inbound'].{name:name, prio:priority, access:access, ports:destinationPortRange, portsList:join(',',destinationPortRanges), src:sourceAddressPrefix}"
done
```

If 80 and 443 are already allowed from `Internet`/`*`, skip to Phase 5. Otherwise, **ASK**, then add
one rule per NSG that lacks them. Choose a priority that is not already in use:

```bash
for NSG in $NIC_NSG $SUBNET_NSG; do
  NAME=$(basename "$NSG"); NRG=$(echo "$NSG" | cut -d/ -f5)
  az network nsg rule create -g "$NRG" --nsg-name "$NAME" -n prism-bio-http-https \
    --priority 1010 --direction Inbound --access Allow --protocol Tcp \
    --source-address-prefixes Internet --destination-port-ranges 80 443
done
```

(Mode B still needs 80/443 open, but they are usually already open because the existing proxy uses
them.)

**Check:** the rule list now shows 80 and 443 allowed inbound.

---

## Phase 5: put the code on the VM

The image is built from the repository root. Prefer cloning on the VM. If the repository is
private, or the clone fails, copy the checkout from the Mac with `rsync` instead.

```bash
# Option 1: clone on the VM (public repo)
ssh "$SSH_TARGET" "sudo mkdir -p /opt/prism-plm && sudo chown \$USER /opt/prism-plm && \
  if [ -d /opt/prism-plm/.git ]; then git -C /opt/prism-plm fetch origin '$DEPLOY_BRANCH' && git -C /opt/prism-plm checkout -B '$DEPLOY_BRANCH' 'origin/$DEPLOY_BRANCH'; \
  else git clone --branch '$DEPLOY_BRANCH' --depth 1 https://github.com/rishimj/prism-plm.git /opt/prism-plm; fi"

# Option 2: rsync from the Mac (private repo or no git on the VM)
git -C /path/to/prism-plm checkout "$DEPLOY_BRANCH" && git -C /path/to/prism-plm pull --ff-only
ssh "$SSH_TARGET" 'sudo mkdir -p /opt/prism-plm && sudo chown $USER /opt/prism-plm'
rsync -az --delete \
  --exclude .git --exclude 'demo/.cache' --exclude 'node_modules' --exclude 'demo/web/dist' --exclude '.venv' \
  /path/to/prism-plm/ "$SSH_TARGET:/opt/prism-plm/"
```

**Check:**
```bash
ssh "$SSH_TARGET" 'ls /opt/prism-plm/demo/backend/assets/ && ls /opt/prism-plm/src/steering/'
```
The assets directory must contain `concepts.json`, `probes.json`, `probe_neurons.json`,
`atlas_labels.json`, `atlas_stats.npz` and `residue_stats.npz`.

---

## Phase 6: install Docker and swap (only if missing)

```bash
ssh "$SSH_TARGET" 'bash -s' <<'EOF'
set -e
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi
docker compose version >/dev/null 2>&1 || sudo docker compose version
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
EOF
```

If Docker was already installed, do not upgrade or reconfigure it; other projects depend on it.

---

## Phase 7: configure and start

Write the environment file (the site origin is `https://rishimj.github.io`; origins never include
a path):

```bash
ssh "$SSH_TARGET" "cat > /opt/prism-plm/demo/deploy/.env" <<EOF
API_DOMAIN=$API_DOMAIN
ALLOWED_ORIGINS=https://rishimj.github.io
RATE_LIMIT_PER_MINUTE=30
TORCH_THREADS=2
MAX_SEQUENCE_LENGTH=400
API_PORT=${API_PORT:-8000}
EOF
```

### Mode A: nothing on 80/443 (bundled Caddy handles TLS)

```bash
ssh "$SSH_TARGET" 'cd /opt/prism-plm/demo/deploy && sudo docker compose up -d --build'
```

The first build downloads PyTorch and the model and takes about 5 to 10 minutes on a B2s.

### Mode B: an existing reverse proxy owns 80/443

Start only the API, bound to localhost:

```bash
ssh "$SSH_TARGET" 'cd /opt/prism-plm/demo/deploy && sudo docker compose -f docker-compose.yml -f docker-compose.behind-proxy.yml up -d --build api'
ssh "$SSH_TARGET" "curl -fsS http://127.0.0.1:${API_PORT:-8000}/api/health"
```

Then add **one new site** for `$API_DOMAIN` to the existing proxy. **ASK** before editing its config.
Back up the config file first, validate it before reloading, and never modify other sites' blocks.

*Existing Caddy (host or container):* append to its Caddyfile, then `caddy validate` and
`caddy reload` (or `docker exec <caddy> caddy reload --config /etc/caddy/Caddyfile`). If Caddy runs
in a container, `127.0.0.1` means the container itself; use `host.docker.internal` (with
`extra_hosts: ["host.docker.internal:host-gateway"]`) or the host's docker bridge IP instead.

```caddyfile
<API_DOMAIN> {
	encode gzip
	reverse_proxy /api/* 127.0.0.1:8000
}
```

*Existing nginx:* create `/etc/nginx/sites-available/prism-bio`, symlink it into `sites-enabled`, run
`sudo nginx -t`, reload, then issue a certificate with
`sudo certbot --nginx -d <API_DOMAIN>` (install `certbot python3-certbot-nginx` if missing; **ASK**
first).

```nginx
server {
    listen 80;
    server_name <API_DOMAIN>;
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
```

The API sets its own CORS headers; do not add CORS headers in the proxy (duplicate headers break
browsers).

---

## Phase 8: verify

Run from the **Mac** (outside the VM):

```bash
# 1. Health over HTTPS with a valid certificate (may take a minute after first start)
for i in $(seq 1 30); do curl -fsS "https://$API_DOMAIN/api/health" && break; sleep 10; done; echo

# 2. CORS preflight from the Pages origin must echo the origin back
curl -si -X OPTIONS "https://$API_DOMAIN/api/analyze" \
  -H 'Origin: https://rishimj.github.io' -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: content-type' | grep -i '^access-control-allow-origin'

# 3. A real inference call
curl -fsS -X POST "https://$API_DOMAIN/api/steer" -H 'Content-Type: application/json' \
  -d '{"sequence":"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG","concept":"zinc_finger","strength":2}' \
  | jq '{identity, concept_score, layer}'
```

Expected: (1) JSON with `"status":"ok"` and `"model":"facebook/esm2_t12_35M_UR50D"`;
(2) `access-control-allow-origin: https://rishimj.github.io`; (3) an identity below 1 and a
numeric concept score.

Troubleshooting:

| Symptom | Likely cause | Look at |
|---|---|---|
| Connection timed out | NSG or OS firewall (`sudo ufw status`) blocks 80/443 | Phase 4 |
| TLS error / certificate not trusted | Caddy could not complete the ACME challenge (port 80 closed, DNS wrong) | `sudo docker compose -p prism-bio logs caddy` |
| 502 from the proxy | API container not up yet or wrong upstream port | `sudo docker compose -p prism-bio logs api`, `docker ps` |
| API container restarting | Out of memory (`dmesg | grep -i oom`) or missing assets | Phase 2 memory, Phase 5 check |
| No `access-control-allow-origin` | `ALLOWED_ORIGINS` wrong or proxy stripping headers | `.env`, Phase 7 note |

---

## Phase 9: connect the GitHub Pages site

Run from the Mac, in the repo:

```bash
# Enable Pages with GitHub Actions as the source (no-op if already enabled)
gh api -X POST repos/rishimj/prism-plm/pages -f build_type=workflow 2>/dev/null \
  || gh api -X PUT repos/rishimj/prism-plm/pages -f build_type=workflow

# Tell the site where the API lives (a repository *variable*, not a secret)
gh variable set DEMO_API_URL --repo rishimj/prism-plm --body "https://$API_DOMAIN"
```

The deploy workflow (`.github/workflows/deploy-demo.yml`) runs only from `main`. If the demo branch
is not merged yet, **ASK** the owner whether to open a pull request
(`gh pr create --base main --head claude/biology-demo-recruiter-52550g`) and stop here until it is
merged. Once it is on `main`:

```bash
gh workflow run deploy-demo.yml --repo rishimj/prism-plm --ref main
sleep 5; gh run watch --repo rishimj/prism-plm $(gh run list --repo rishimj/prism-plm --workflow deploy-demo.yml -L1 --json databaseId -q '.[0].databaseId')
```

**Check:** open `https://rishimj.github.io/prism-plm/`. The header pill must read **"Live model
online"** (green). Before the rebuild finishes you can test with
`https://rishimj.github.io/prism-plm/?api=https://$API_DOMAIN`.

---

## Phase 10: report back to the owner

Summarize: the VM and mode used, `API_DOMAIN`, NSG rules added (names), any proxy config files
created or edited (with backup paths), the results of the three Phase 8 checks, and whether the
site shows "Live model online".

---

## Operations reference

```bash
# Update to the latest code
ssh "$SSH_TARGET" "cd /opt/prism-plm && git pull --ff-only && cd demo/deploy && sudo docker compose up -d --build"
#   (rsync mode: re-run the Phase 5 rsync, then the same docker compose command)
#   Mode B: add -f docker-compose.yml -f docker-compose.behind-proxy.yml ... api

# Logs / status
ssh "$SSH_TARGET" "sudo docker compose -p prism-bio ps; sudo docker compose -p prism-bio logs --tail 100 api"

# Stop (keeps images and the TLS certificate volume)
ssh "$SSH_TARGET" "sudo docker compose -p prism-bio down"

# Remove everything this runbook created
ssh "$SSH_TARGET" "sudo docker compose -p prism-bio down -v --rmi local && sudo rm -rf /opt/prism-plm"
# NSG rule(s): az network nsg rule delete -g <nsg-rg> --nsg-name <nsg> -n prism-bio-http-https
# Mode B: remove the prism-bio site block / nginx site and reload the proxy
# The swap file is harmless to keep; remove only if the owner asks.
```

When the VM is off, the site keeps working on precomputed data and the pill shows
"Precomputed mode", so stopping the API is always safe.
