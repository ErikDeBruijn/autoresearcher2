---
name: research-management
description: Manage AutoResearcher2 projects, proposals, and research pipeline. Use when users ask about creating projects, submitting proposals, checking experiment status, or managing the research pipeline.
---

# Research Management

Manage the AutoResearcher2 research pipeline through CLI tools.

## Available Commands

All commands talk to the local AutoResearcher2 API at `http://localhost:8000`.

### Create a project

```bash
research-create-project --name "Project Name" --description "What we optimize" \
  --domain nanogpt \
  --parameters "DEPTH, MATRIX_LR, WEIGHT_DECAY"
```

Domains: `nanogpt`, `atari-rl`, `generic`

### List projects

```bash
research-list-projects
```

### Submit a proposal

```bash
research-submit-proposal --project proj_abc \
  --intent "Test whether higher LR improves val_bpb" \
  --type config_change \
  --spec '{"MATRIX_LR": "0.04"}'
```

### Get pipeline status

```bash
research-status
```

## When to Use

- User asks to **create a project** → gather name, description, domain, parameters → `research-create-project`
- User asks about **current status** → `research-status`
- User wants to **add an experiment** → `research-submit-proposal`
- User asks to **list projects** → `research-list-projects`

## Conversation Flow for Project Creation

1. Ask what the user wants to optimize (goal, metric)
2. Ask what parameters can be varied
3. Determine the domain type (nanogpt/atari-rl/generic)
4. Call `research-create-project` with the gathered info
5. Report the created project ID to the user
