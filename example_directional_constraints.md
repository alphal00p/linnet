# Directional Constraint Groups

This document demonstrates the new directional constraint group feature that allows restricting shifts to only positive or negative directions.

## Syntax

### Basic Constraint Groups (existing)
- `pin="@group1"` - Links both x and y coordinates to group1, allows shifts in any direction
- `pin="x:@group1"` - Links only x coordinate to group1
- `pin="y:@group1"` - Links only y coordinate to group1

### Directional Constraint Groups (new)
- `pin="@+group1"` - Links both coordinates to group1, only allows positive shifts
- `pin="@-group1"` - Links both coordinates to group1, only allows negative shifts
- `pin="x:@+group2"` - Links x coordinate to group2, only allows positive x shifts
- `pin="y:@-group3"` - Links y coordinate to group3, only allows negative y shifts

## How It Works

When a constraint group has a leading `+` or `-` prefix:
- `+group`: Only positive shifts are allowed (negative shifts are clamped to 0)
- `-group`: Only negative shifts are allowed (positive shifts are clamped to 0)
- No prefix: Any shifts are allowed (existing behavior)

## Examples

### Example 1: Positive-only horizontal movement
```
node1 [pin="x:@+horizontal"];
node2 [pin="x:@+horizontal"];
node3 [pin="x:@+horizontal"];
```
All three nodes are
