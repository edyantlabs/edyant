# Edyant Framework

A framework for AI systems that maintain meaningful and ethical continuity across interactions, contexts, and time.

This project treats memory as lived contextual experience so agents can retain skills, world models, and performance history across sessions. 
The goal is to enable safe, reliable long-term collaboration between humans and AI in shared organizational and personal environments.

## What it provides

- Layered memory for episodic, semantic, and procedural experience
- Skill retention across sessions and multi-session task resumption
- Performance awareness, drift/degradation monitoring, and adaptive compensation
- Safety-first incident memory, explainable recall, and governance-aware retention

## Technical details

| Field              | Value                                                                                                           |
|--------------------|-----------------------------------------------------------------------------------------------------------------|
| Hardware           | CPU-first; no GPU or CUDA required                                                                              |
| Acceleration       | Optional (hardware-agnostic)                                                                                    |
| Project Name       | `edyant`                                                                                            |
| Description        | Framework for AI systems that maintain continuity across interactions, contexts, and time                       |
| Python Requirement | Python `>= 3.11`                                                                                                |
| License            | [Apache License 2.0](LICENSE)                                                                                   |
| Author             | [**Edyant Labs**](labs.edyant.com)                                                                              |
| Contact            | arsalan@edyant.com                                                                                              |

## Install

```bash
python -m pip install edyant
```

Test releases (TestPyPI):

```bash
python -m pip install \
  --index-url https://pypi.org/simple \
  --extra-index-url https://test.pypi.org/simple/ \
  edyant
```

## How users will import

### Complete framework:
```python
import edyant
```

### Specific subsystems:

```python
from edyant import benchmark
from edyant import core
from edyant import ethics
from edyant import persistence
```

> **edyant.persona** and **edyant.umwelt** on hiatus for now to focus on benchmark and ethics.

## Docs and License

See [docs/](docs/) for documentation, design docs, and research threads
The license is in [LICENSE](LICENSE).
