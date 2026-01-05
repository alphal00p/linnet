# Agent Working Notes

This file contains ongoing notes, observations, and context from agent sessions working on the Linnet project.

## Current Session

**Date:** 2024-12-19
**Focus:** Setting up agent context management for Linnet project
**Status:** Active

### Key Observations
- Project uses jj (Jujutsu) for version control, not git
- User prefers simple, readable markdown over complex JSON schemas
- Working in Zed IDE environment
- Half-edge data structure is central to everything

### Important Discoveries
- All algorithms work at subgraph level (whole graph is special case)
- Tensor networks and Feynman diagrams are primary use cases
- Related to gammaloop and spenso projects
- Performance-critical - used in computational physics

### Current Understanding
- Version 0.14.1, Rust, MIT license
- Author: Lucien Huber
- Core innovation: half-edge structure for clean graph splitting
- Build system uses `just` command runner

## Previous Sessions

### Session Template
```
**Date:** YYYY-MM-DD
**Focus:** Brief description
**Status:** Complete/Ongoing/Paused

#### What was accomplished:
- Item 1
- Item 2

#### Key insights:
- Insight 1
- Insight 2

#### Next steps:
- Step 1
- Step 2

#### Files modified:
- file1.rs
- file2.md
```

## Ongoing Context

### Half-Edge Structure Notes
- Enables efficient subgraph manipulation
- Preserves node degrees during graph splitting
- All operations designed around this concept
- Performance optimized for frequent subgraph ops

### Development Patterns
- Use `just build --features serde,drawing` for development
- Test with `just test` 
- Format with `just fmt`
- Lint with `just clippy`

### Common Issues to Watch For
- Half-edge pairing consistency
- Subgraph boundary integrity  
- Performance degradation with large subgraphs
- Memory usage in iterative operations

### Architecture Decisions
- Subgraph-first design philosophy
- Half-edge as core abstraction
- Physics applications drive requirements
- Performance critical for computational use

## Questions & TODOs

### Current Questions
- How are tensor contractions mapped to graph operations?
- What's the relationship with gammaloop specifically?
- Performance bottlenecks in current implementation?

### Future Investigation
- [ ] Explore src/ directory structure
- [ ] Understand benchmarking setup in benches/
- [ ] Review examples/ for usage patterns
- [ ] Check integration with gammaloop/spenso

## Quick References

### Build Commands
```bash
just build          # Build all packages
just test           # Run tests
just bench          # Run benchmarks
just fmt            # Format code
just clippy         # Lint code
```

### Key Features
- `serde` - Serialization
- `drawing` - Visualization 
- `symbolica` - Symbolic math
- `bincode` - Binary serialization

### Project Layout
```
src/        - Main library
clinnet/    - CLI tool
linnest/    - Testing utilities
examples/   - Usage examples
benches/    - Performance tests
```

---

*Keep this file updated with each session. Focus on actionable insights and maintaining continuity between sessions.*
