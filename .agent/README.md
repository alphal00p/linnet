# Linnet Agent Knowledge Base

This file contains key context and knowledge about the Linnet project for AI agents working with the codebase.

## Project Overview

**Linnet** is a Rust graph library specifically designed for **tensor networks** and **Feynman diagrams**. The core innovation is using a **half-edge data structure** that enables efficient subgraph manipulation.

- **Current Version**: 0.14.1
- **Language**: Rust
- **License**: MIT
- **Author**: Lucien Huber
- **Repository**: Uses `jj` (Jujutsu) for version control

## Core Concepts

### Half-Edge Data Structure
- **Central Design Choice**: All operations built around half-edge representation
- **Key Benefit**: Enables clean splitting of graphs while preserving node degrees
- **Operations**: Edge contraction, node identification, subgraph excision, graph joining

### Subgraph-Centric Design
- **Important**: ALL algorithms operate at the subgraph level
- The whole graph is just a special case of a subgraph
- This design choice permeates the entire codebase

### Primary Applications
1. **Tensor Networks**: Mathematical structures used in quantum physics and ML
2. **Feynman Diagrams**: Particle interaction representations in quantum field theory
3. **Related Projects**: `gammaloop` and `spenso` depend on Linnet

## Project Structure

```
linnet/
├── src/                    # Main library code
├── clinnet/               # Command-line interface
├── linnest/               # Testing utilities
├── examples/              # Usage examples
├── benches/               # Performance benchmarks
├── Cargo.toml             # Dependencies and features
├── Justfile               # Build automation
└── README.md              # Project documentation
```

## Key Dependencies

### Core Dependencies
- `ahash` - Fast hashing
- `bitvec` - Efficient bit vectors
- `indexmap` - Ordered hashmaps
- `itertools` - Iterator utilities
- `rand` - Random number generation

### Optional Features
- `serde` - Serialization support
- `drawing` - Graph visualization (cgmath, frostfire)
- `symbolica` - Symbolic computation
- `bincode` - Binary serialization

## Development Workflow

### Build System
- Uses `just` for task automation
- Primary commands:
  - `just build` - Build all packages
  - `just test` - Run tests
  - `just bench` - Run benchmarks
  - `just fmt` - Format code
  - `just clippy` - Lint code

### Features to Enable
Most development should use: `cargo build --features serde,drawing`

### Testing Strategy
- Main tests in `cargo test -p linnet --features serde`
- Separate linnest package for specialized testing
- Benchmark suite for performance monitoring

## Common Patterns

### Graph Creation
```rust
// Half-edge graphs with subgraph operations
// Focus on subgraph extraction and manipulation
```

### Algorithm Implementation
- Design algorithms to work on subgraphs
- Consider half-edge structure in all operations
- Maintain efficiency for frequent subgraph operations

### Visualization
- Built-in graph drawing capabilities
- Simulated annealing for layout optimization
- SVG output support

## Important Context for Agents

### What Makes Linnet Special
1. **Half-edge focus**: Not just another graph library
2. **Subgraph operations**: Core design principle affects everything
3. **Physics applications**: Understanding tensor networks and Feynman diagrams helps
4. **Performance critical**: Used in computational physics applications

### Common Questions
- **"How do I create a graph?"** → Focus on half-edge structure and subgraph definition
- **"Why this design?"** → Enables clean graph splitting while preserving node degrees
- **"Performance concerns?"** → Subgraph operations are optimized, whole-graph operations may be slower

### Debugging Tips
- Check half-edge pairing consistency
- Verify subgraph boundary integrity
- Monitor subgraph size for performance issues

### Integration Points
- Works with `gammaloop` for Feynman diagram operations
- Works with `spenso` for tensor network operations
- Dot format parsing for graph input

## Development Environment

### Recommended Setup
- Use `jj` for version control (project uses jujutsu)
- Rust with clippy and rustfmt
- Features: `serde`, `drawing` for full functionality

### Performance Monitoring
- Benchmark suite in `benches/`
- Focus on subgraph operation performance
- Memory usage important for large graphs

## Recent Changes & Evolution

Check `CHANGELOG.md` and recent commits for:
- API changes affecting subgraph operations
- Performance improvements
- New algorithms or features
- Breaking changes in half-edge handling

## Quick Reference

### File Extensions & Formats
- `.rs` - Rust source code
- `.toml` - Configuration (Cargo.toml, etc.)
- `.just` - Justfile recipes
- `.md` - Documentation

### Key Modules (likely in src/)
- Graph core functionality
- Half-edge operations
- Subgraph algorithms
- Visualization/drawing
- Serialization support

---

*This knowledge base should be updated as the project evolves. Focus on half-edge operations and subgraph thinking when working with this codebase.*
