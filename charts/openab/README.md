# openab Helm Chart

This chart deploys one or more OpenAB agents on Kubernetes.

## Common Values

This page highlights commonly used values and deployment patterns. For the complete list of supported options and defaults, run `helm show values openab/openab` or inspect [`values.yaml`](values.yaml).

### Release naming

| Value | Description | Default |
|-------|-------------|---------|
| `nameOverride` | Override the chart name portion used in generated resource names. For per-agent resource names, use `agents.<name>.nameOverride`. | `""` |
| `fullnameOverride` | Override the full generated release name for chart resources. Useful when deploying multiple instances with predictable names. | `""` |
| `serviceAccountName` | Chart-global ServiceAccount name attached to every agent pod that doesn't define its own. Empty = cluster `default` SA. Per-agent `agents.<name>.serviceAccountName` fully overrides this. Chart references an existing SA only — does not create one. Required for workload identity and pod-level RBAC. | `""` |
| `imagePullSecrets` | Chart-global image pull secrets attached to every agent pod that doesn't define its own. Per-agent `agents.<name>.imagePullSecrets` fully overrides this. | `[]` |
| `networkPolicy.enabled` | Master switch for the chart-managed NetworkPolicy. Off by default — no `NetworkPolicy` resource is rendered until this flips to `true`, keeping existing releases unaffected. When on: one policy per agent (deny-all ingress + DNS/HTTPS egress + TCP 8080 to the chart-managed gateway pod when `gateway.enabled=true` and `gateway.deploy` is not `false`), plus one policy per deployed gateway (port 8080 ingress + same egress). ⚠️ **Prerequisite: your cluster's CNI must enforce NetworkPolicy** (Calico, Cilium, Antrea, EKS with NetworkPolicy add-on, GKE with Dataplane V2 / Calico, AKS with Azure NPM / Calico, etc.). Stock KIND, Docker Desktop, or bare-EKS clusters without a NetworkPolicy-capable CNI silently ignore the resource — verify enforcement with a scratch namespace + deny-all test before relying on this. ⚠️ **Direct egress only for now** — proxy-mode egress (routing through an operator-supplied HTTP CONNECT proxy) was explored in early rounds of [RFC #1394](https://github.com/openabdev/openab/issues/1394) but requires a proxy-aware networking refactor across `openab-core` / `openab-gateway` (see `docs/openshell.md`). Follow-up RFC to come. | `false` |
| `networkPolicy.ingress.extraRules` | Raw `NetworkPolicyIngressRule` entries appended verbatim to every rendered policy. Use for cluster-specific allow-lists (Prometheus scrape, mesh sidecars, etc.). | `[]` |
| `networkPolicy.egress.allowDns` | Allow DNS egress on UDP+TCP 53 to pods labelled `k8s-app=kube-dns`. | `true` |
| `networkPolicy.egress.dnsNamespace` | Trusted namespace hosting the DNS pods. Default `"kube-system"` matches kubeadm / CoreDNS / kube-dns defaults and prevents a co-tenant labelling their own pod `k8s-app=kube-dns` from becoming an approved port-53 destination. Empty string reverts to the legacy any-namespace selector — use only if your DNS runs outside kube-system. | `"kube-system"` |
| `networkPolicy.egress.allowHttps` | Allow HTTPS (TCP 443) to `0.0.0.0/0` (and `::/0` when `allowIpv6=true`), except CIDRs listed in `metadataExclusions`. | `true` |
| `networkPolicy.egress.allowHttp` | Allow HTTP (TCP 80) with the same metadata exclusion. Off by default. ⚠️ Enable if any of these fetch over HTTP: `agents.<name>.configUrl`, lifecycle hooks that download over HTTP (see `docs/hooks.md`), or STT base URLs on HTTP. Prefer HTTPS wherever possible. | `false` |
| `networkPolicy.egress.allowIpv6` | Add `::/0` to the HTTPS/HTTP rules alongside `0.0.0.0/0`, excluding CIDRs from `metadataExclusions.ipv6`. Turn off on IPv4-only clusters. | `true` |
| `networkPolicy.egress.metadataExclusions.ipv4` | CIDRs excluded from the IPv4 HTTPS/HTTP allow rule. Defaults to `169.254.169.254/32` (AWS/GCP/Azure IMDS). Add IBM Cloud's `161.26.0.0/16` here if applicable. **NetworkPolicy allow rules combine additively — you cannot subtract via `extraRules`; edit this list directly.** ⚠️ **Defense-in-depth only, not a portable IMDS guarantee**: pods can always reach services on their resident node, and CNI implementations differ in whether `ipBlock` matches pre- or post-NAT addresses. Use cloud-provider hardening (IMDSv2 mandatory + hop-limit 1, GCP metadata concealment, Azure metadata endpoint policy) as the primary control. | `["169.254.169.254/32"]` |
| `networkPolicy.egress.metadataExclusions.ipv6` | CIDRs excluded from the IPv6 HTTPS/HTTP allow rule. Defaults to `fd00:ec2::254/128` (AWS IPv6 IMDS). Non-AWS IPv6 metadata addresses differ — override this list per cloud. **Same additive-only semantics as `ipv4` above, and same defense-in-depth caveat.** | `["fd00:ec2::254/128"]` |
| `networkPolicy.egress.extraRules` | Raw `NetworkPolicyEgressRule` entries appended verbatim. Use to open extra ports or add cluster-specific carve-outs. **Cannot subtract from existing allows — for tighter metadata exclusions use `metadataExclusions` above.** ⚠️ **Non-80/443 destinations require an explicit entry here** — enabling `networkPolicy.enabled=true` silently breaks: (a) external config-only gateway (`gateway.deploy=false` + URL supplied via `configToml`'s `[gateway]` section on TCP 8080), (b) self-hosted STT on non-standard ports (e.g. `http://192.168.1.100:8080/v1`), (c) internal LLM proxies (LiteLLM on 4000, etc.). See `values.yaml` for a complete `extraRules` + `configToml` recipe for the external-gateway case. | `[]` |

### Agent values

Each agent lives under `agents.<name>`.

| Value | Description | Default |
|-------|-------------|---------|
| `discord.botToken` | Discord bot token for the agent. | `""` |
| `discord.allowedChannels` | Channel allowlist. Use `--set-string` for Discord IDs. | `["YOUR_CHANNEL_ID"]` |
| `discord.allowedUsers` | User allowlist. Empty = allow all users by default. Use `--set-string` for Discord IDs. | `[]` |
| `discord.allowDm` | Whether the Discord bot responds to direct messages. | `false` |
| `discord.allowBotMessages` | Controls whether bot messages can trigger replies. | `"off"` |
| `discord.trustedBotIds` | Optional bot ID allowlist when bot-message replies are enabled. | `[]` |
| `slack.enabled` | Enable the Slack adapter for the agent. | `false` |
| `slack.botToken` | Slack Bot User OAuth token. | `""` |
| `slack.appToken` | Slack App-Level token for Socket Mode. | `""` |
| `slack.existingSecret` | Name of a pre-existing K8s Secret containing `slack-bot-token` and `slack-app-token`. When set, `botToken`/`appToken` above are ignored and the chart skips creating those keys. Enables External Secrets Operator / Vault / SealedSecrets workflows. | `""` |
| `slack.allowedChannels` | Slack channel allowlist. Empty means allow all channels by default. | `[]` |
| `slack.allowedUsers` | Slack user allowlist. Empty means allow all users by default. | `[]` |
| `nameOverride` | Override this agent's generated resource name. | `""` |
| `workingDir` | Working directory and HOME inside the container. | `"/home/agent"` |
| `env` | Inline environment variables passed to the agent process. | `{}` |
| `envFrom` | Additional environment sources from existing Secrets or ConfigMaps. | `[]` |
| `pool.maxSessions` | Maximum concurrent ACP sessions for the agent. | `10` |
| `pool.sessionTtlHours` | Idle session TTL in hours. | `24` |
| `reactions.enabled` | Enable status reactions. | `true` |
| `reactions.removeAfterReply` | Remove status reactions after the agent replies. | `false` |
| `reactions.toolDisplay` | Tool display verbosity: `full`, `compact`, or `none`. | `"full"` |
| `stt.enabled` | Enable voice-message speech-to-text. | `false` |
| `stt.apiKey` | API key for the speech-to-text provider. | `""` |
| `stt.model` | STT model name. | `"whisper-large-v3-turbo"` |
| `stt.baseUrl` | STT API base URL. | `"https://api.groq.com/openai/v1"` |
| `gateway.enabled` | Enable the gateway config block for webhook-based platforms. | `false` |
| `gateway.deploy` | Deploy the gateway Deployment and Service. ⚠️ When `false` (config-only / external gateway), you must (a) supply the gateway URL via `configToml` (a `[gateway]` section — the standalone `gateway.url` value is not consumed by the ConfigMap since chart v0.10), and (b) if `networkPolicy.enabled=true`, add a matching `networkPolicy.egress.extraRules` entry targeting your external gateway pod/service, otherwise the agent→external-gateway WSS is blocked. See `values.yaml` for the complete recipe. | `true` |
| `gateway.env` | Additional environment variables for the gateway container (map, `{name: value}`). Same shape as agent-level `env`. Useful for feature flags, `RUST_LOG` overrides, or HTTP proxy env vars if the gateway runs behind a transparent proxy. | `{}` |
| `gateway.envFrom` | Load env vars for the gateway from existing Secrets or ConfigMaps (list of `secretRef`/`configMapRef` entries). Same shape as agent-level `envFrom`. | `[]` |
| `cron.usercronEnabled` | Enable user-provided cron configuration. | `false` |
| `cronjobs` | Config-driven scheduled messages for an agent. | `[]` |
| `persistence.enabled` | Enable persistent storage for auth and settings. | `true` |
| `persistence.existingClaim` | Reuse an existing PVC instead of creating one. | `""` |
| `agentsMd` | Contents of `AGENTS.md` mounted into the working directory. | `""` |
| `serviceAccountName` | Per-agent ServiceAccount name. When set (non-empty), fully overrides chart-global `serviceAccountName`. Useful when only some agents need a dedicated SA. | `""` |
| `imagePullSecrets` | Per-agent image pull secrets. When set, fully overrides chart-global `imagePullSecrets`. Useful when only some agents pull from a private registry. | `[]` |
| `extraInitContainers` | Additional init containers for the agent pod. | `[]` |
| `extraContainers` | Additional sidecar containers for the agent pod. | `[]` |
| `extraVolumeMounts` | Additional volume mounts for the main agent container. | `[]` |
| `extraVolumes` | Additional volumes for the agent pod. | `[]` |

## Examples

### Override generated names

```bash
helm install prod openab/openab \
  --set fullnameOverride=my-openab \
  --set-literal agents.kiro.discord.botToken="$DISCORD_BOT_TOKEN" \
  --set-string 'agents.kiro.discord.allowedChannels[0]=YOUR_CHANNEL_ID'
```

This makes generated resource names use `my-openab` (for example `my-openab-kiro`) instead of the default `prod-openab`.

### Load credentials with `envFrom`

```yaml
agents:
  kiro:
    envFrom:
      - secretRef:
          name: openab-agent-secrets
      - configMapRef:
          name: openab-agent-config
```

This is useful for credentials such as `GH_TOKEN` without storing them directly in Helm values.

### Provide `AGENTS.md` with `--set-file`

```bash
helm install openab openab/openab \
  --set-literal agents.kiro.discord.botToken="$DISCORD_BOT_TOKEN" \
  --set-string 'agents.kiro.discord.allowedChannels[0]=YOUR_CHANNEL_ID' \
  --set-file agents.kiro.agentsMd=./AGENTS.md
```

### Provide `config.toml` as-is with `--set-file`

`configToml` accepts a raw TOML string, which can be pasted inline into `values.yaml`
or loaded verbatim from a standalone file. Keeping `config.toml` as a real file gives
you full IDE syntax highlighting and TOML schema validation, instead of an indented
YAML block scalar:

```bash
helm upgrade openab openab/openab \
  --set-file agents.kiro.configToml=./config.toml
```

See [`docs/migrate-to-configtoml.md`](../../docs/migrate-to-configtoml.md) for a full before/after guide, and
[`docs/adr/configurl-over-helm-rendering.md`](../../docs/adr/configurl-over-helm-rendering.md) for when to prefer `configUrl` instead
(platform-agnostic — works identically on Kubernetes, ECS, Zeabur, and AgentCore).

### Discord ID precision warning

Discord IDs must be set with `--set-string`, not `--set`. Otherwise Helm may coerce them into numbers and lose precision.
