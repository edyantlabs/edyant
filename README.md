# Persistence & Continuity Framework

A framework for AI systems that maintain meaningful continuity across interactions, contexts, and time.

This project treats memory as lived contextual experience so agents can retain skills, world models, and performance history across sessions. The goal is to enable safe, reliable long-term collaboration between humans and AI in shared organizational and personal environments.

## What it provides

- Layered memory for episodic, semantic, and procedural experience
- Skill retention across sessions and multi-session task resumption
- Performance awareness, drift/degradation monitoring, and adaptive compensation
- Safety-first incident memory, explainable recall, and governance-aware retention

## Status

Early research and framework design. APIs and storage formats are expected to change.

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
from edyant import persona
from edyant import umwelt
```

## Learn more

See [about.md](docs/about_persistence.md) for the full framework narrative, research threads, and governance considerations. The license is in [LICENSE](LICENSE).
