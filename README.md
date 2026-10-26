# Counterfeit Detection | C++17 Product-Serial Verification

A small, testable C++17 verification engine that compares submitted product serials and metadata against a local registry. It returns structured reason codes for malformed, unknown, duplicate, or mismatched scans. The data is synthetic and the design is deliberately explicit about what a registry match proves—and what it does not.

## Features
- Separated verification core and CLI; no third-party C++ dependencies.
- Input format checks, registry lookup, metadata comparison, and per-process duplicate detection.
- JSON-shaped CLI output and CMake/CTest tests.
- Synthetic registry data suitable for demos only.

## Build and run
```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/counterfeit-check data/registry.csv ABCD-10001 Headphones DemoCo
```
Expected success code: `REGISTRY_MATCH`. Re-running the same serial within one process would be a duplicate, but each CLI invocation starts a fresh process; use the core in a long-running service to demonstrate duplicate tracking across requests.

## Decision codes
| Code | Meaning |
|---|---|
| `REGISTRY_MATCH` | Serial and submitted metadata match the local registry |
| `UNKNOWN_SERIAL` | Serial not found |
| `DUPLICATE_SCAN` | Serial already accepted by this verifier instance |
| `METADATA_MISMATCH` | Product or manufacturer differs from registry |
| `INVALID_FORMAT` | Serial violates length or allowed-character rules |

## Security and scope
This is a rule-based registry prototype, not an RNN model and not proof a physical item is genuine. The local CSV can be altered, serials can be copied, and duplicate tracking is in-memory. A real deployment should use authenticated manufacturer data, signed/cryptographically random identifiers, a persistent transactional database, rate limiting, audit logs, and a threat model. Do not claim a model accuracy or Heroku deployment for this implementation.
