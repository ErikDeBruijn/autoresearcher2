# Project Priority System

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let users control which projects get more research attention, with an Auto mode that uses expected_gain (learntropy-based) to allocate resources dynamically.

**Architecture:** Add `priority` field to projects table. Planner adjusts n_proposals/n_select per project based on priority. Frontend shows dropdown. No changes to worker or claim logic — priority is implemented through queue density.

**Tech Stack:** SQLite (store.py), FastAPI (api.py), React/Next.js (ProjectFilter.tsx)

---

## Priority Levels

| Value | Label | n_proposals | n_select | Behavior |
|-------|-------|-------------|----------|----------|
| `exclusive` | Exclusive | 5 | 3 | Only this project runs. Others skip planner + claim. |
| `high` | High | 5 | 3 | Full allocation |
| `normal` | Normal | 3 | 2 | Standard |
| `low` | Low | 2 | 1 | Minimal |
| `paused` | Paused | 0 | 0 | No new proposals, workers skip |
| `auto` | Auto | computed | computed | Based on expected_gain |

## Auto Priority: Expected Gain

```python
def compute_expected_gain(store, project_id) -> float:
    """Returns 0.0-1.0 score indicating how much learning potential remains."""
    wm = store.load_world_model(project_id=project_id)
    history = store.list_world_model_history(project_id=project_id)

    # 1. Recent learntropy trend (last 5 updates)
    recent = [h["delta"].get("learntropy", 0) for h in history[-5:]]
    avg_learntropy = sum(recent) / len(recent) if recent else 0.5

    # 2. Low-confidence belief ratio
    beliefs = wm.beliefs
    low_conf = [b for b in beliefs if b.get("confidence", 1.0) < 0.5]
    uncertainty_ratio = len(low_conf) / max(len(beliefs), 1)

    # 3. Unresolved tension ratio
    tension_ratio = len(wm.tensions) / max(len(beliefs), 1)

    # Weighted combination
    return min(1.0, 0.4 * avg_learntropy + 0.35 * uncertainty_ratio + 0.25 * tension_ratio)
```

Auto maps expected_gain to slots:
- gain >= 0.5 → high (5 proposals, 3 select)
- gain >= 0.2 → normal (3 proposals, 2 select)
- gain < 0.2 → low (2 proposals, 1 select)

## Changes

### 1. DB Schema (store.py)
Add column: `ALTER TABLE projects ADD COLUMN priority TEXT DEFAULT 'auto'`

### 2. Store (store.py)
- `create_project()`: accept `priority` param
- `update_project()`: already supports kwargs, priority comes free
- `list_projects()`: return priority field
- New: `compute_expected_gain(project_id)` method

### 3. Planner loop (run_v4_real.py)
```python
def get_priority_slots(store, project):
    priority = project.get("priority", "auto")
    if priority == "paused":
        return None  # skip
    if priority == "exclusive":
        return {"n_proposals": 5, "n_select": 3}
    if priority == "high":
        return {"n_proposals": 5, "n_select": 3}
    if priority == "normal":
        return {"n_proposals": 3, "n_select": 2}
    if priority == "low":
        return {"n_proposals": 2, "n_select": 1}
    # auto
    gain = compute_expected_gain(store, project["id"])
    if gain >= 0.5:
        return {"n_proposals": 5, "n_select": 3}
    if gain >= 0.2:
        return {"n_proposals": 3, "n_select": 2}
    return {"n_proposals": 2, "n_select": 1}
```

Exclusive mode: if any project is exclusive, skip all others in planner loop AND pass only that project's ID to workers.

### 4. API (api.py)
- `GET /api/projects`: already returns all fields, priority comes free
- `PATCH /api/projects/{id}`: already accepts kwargs
- New: `GET /api/projects/{id}/expected_gain` → returns computed gain (for UI display)

### 5. Frontend (ProjectFilter.tsx)
Replace play/pause buttons with a priority dropdown:
- Dropdown options: Auto, Exclusive, High, Normal, Low, Paused
- Show computed expected_gain next to "Auto" when selected
- Color-coded: Exclusive=red, High=yellow, Normal=green, Low=gray, Paused=gray-dim, Auto=blue

### 6. Worker claim filtering
`claim_next_todo()` already filters on `p.active = 1`. Map priority:
- `paused` → active=0
- Everything else → active=1
- `exclusive` → workers receive only this project_id in their filter list
