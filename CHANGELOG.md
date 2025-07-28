# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0](https://github.com/alphal00p/linnet/compare/v0.8.0...v0.9.0) - 2025-07-27

### Other

- Allow specifying a subgraph in the DOT format.
- Rewrite DOT output to include node, hedge and edge indices and fix bidirectional serialization
- DotGraph has concrete data. Parser now supports global edge and vertex data. Can set edge, vertex and hedge index
- add multi-graph parsing and global graph data

## [0.8.0](https://github.com/alphal00p/linnet/compare/v0.7.1...v0.8.0) - 2025-07-14

### Other

- clippy fix
- Rename HedgeVec to EdgeVec and introduce HedgeVec
- Add H parameter to HedgeGraph in iter_edges and orientation
- add H to SignedCycle
- rename as_ref to to_ref
- Add hedge_data to hedge_graph
- add perm by key
- update symbolica
- Fix tree traversal and modify ForestNodeStore API to use Option

## [0.7.1](https://github.com/alphal00p/linnet/compare/v0.7.0...v0.7.1) - 2025-05-24

### Other

- Fix crown computation by inverting subgraph inclusion check and add inclusion for HedgePair

## [0.7.0](https://github.com/alphal00p/linnet/compare/v0.6.3...v0.7.0) - 2025-05-23

### Other

- unify iterator signature and remove redundant and misleading functions
- Add zenodo doi

## [0.6.3](https://github.com/alphal00p/linnet/compare/v0.6.2...v0.6.3) - 2025-05-23

### Other

- improve README

## [0.6.2](https://github.com/alphal00p/linnet/compare/v0.6.1...v0.6.2) - 2025-05-22

### Other

- remove useless check and set
- Correctly forget identification history and adds subgraph contraction

## [0.6.1](https://github.com/alphal00p/linnet/compare/v0.6.0...v0.6.1) - 2025-05-22

### Other

- Merge pull request #11 from alphal00p/jules_wip_3253120278989361716

## [0.6.0](https://github.com/alphal00p/linnet/compare/v0.5.2...v0.6.0) - 2025-05-22

### Fixed

- fix all tests and update symbolica
- fixes forget_identification_history, If you delete a subgraph you need

### Other

- Add crown functions and implement spanning tree enumeration algo based on graph contraction
- PartialEq and Eq for hedgevec
- from_raw for hedgevec

## [0.5.2](https://github.com/alphal00p/linnet/compare/v0.5.1...v0.5.2) - 2025-05-09

### Fixed

- fix involution extraction

### Other

- Add Permutation iterators, new graph extract test

## [0.5.1](https://github.com/alphal00p/linnet/compare/v0.5.0...v0.5.1) - 2025-05-05

### Other

- Clean up debug prints and add cycle test

## [0.5.0](https://github.com/alphal00p/linnet/compare/v0.4.0...v0.5.0) - 2025-05-05

### Fixed

- fixes all around

### Other

- boundary->crown
- hairs->boundary

## [0.4.0](https://github.com/alphal00p/linnet/compare/v0.3.0...v0.4.0) - 2025-04-30

### Added

- feature flag bincode and serde

## [0.3.0](https://github.com/alphal00p/linnet/compare/v0.2.3...v0.3.0) - 2025-04-30

### Fixed

- fixed extraction for childvec forest, all tests pass and fix warnings

### Other

- adds extract to SmartHedgeVec and involution
- maps now take a FnMut 
- derive more (Decode and Encode)

## [0.2.3](https://github.com/alphal00p/linnet/compare/v0.2.2...v0.2.3) - 2025-04-08

### Other

- unpin serde

## [0.2.2](https://github.com/alphal00p/linnet/compare/v0.2.1...v0.2.2) - 2025-04-01

### Fixed

- fix test snapshots

### Other

- raw image

## [0.2.1](https://github.com/alphal00p/linnet/compare/v0.2.0...v0.2.1) - 2025-04-01

### Other

- Update README.md, logo is a github asset
- put crates badge

## [0.2.0](https://github.com/alphal00p/linnet/compare/v0.1.0...v0.2.0) - 2025-04-01

### Fixed

- fix cetz orientation

### Other

- Create release-plz.yml
