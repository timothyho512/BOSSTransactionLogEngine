# BOSSTransactionLogEngine Implementation Log

## Current stage

This folder contains the current lazy transaction log engine prototype implemented in BOSS.

## Main implemented ideas

- The transaction log engine records updates in a WAL-style structure instead of applying every update immediately.
- Updates are grouped by row so that multiple deferred writes to the same row can be folded before materialisation.
- The engine tracks update sequence numbers so that deferred writes can be applied in the correct logical order.
- Reads force materialisation of any pending updates needed to return the correct value.
- A WAL threshold controls when accumulated deferred updates are flushed to the in-memory engine.
- Blind writes can skip earlier deferred work when it is safe to do so.
- Instrumentation counters were added to measure WAL pushes, flushes, folded writes and physical writes.

## Current limitations / postponed work

- Same-row different-column dependencies have been left for later investigation.
- Different-row different-column dependencies have also been left for later investigation.
- The current priority is profiling the existing implementation before adding more dependency cases.
- The implementation is still experimental and may be iterated after profiling.

## Reason for snapshot

This version is being saved as a reference point before further changes. It records the first substantial lazy transaction log implementation, so later report sections can explain what was implemented, what was measured and why the next stage focused on profiling.
