# Dialogue VA Pipeline: Technical Reference

This document describes the implementation, data flow, configuration model,
cache behavior, and alignment rules of Dialogue VA Pipeline. For installation
and normal operation, see [README.md](README.md).

## Pipeline and artifacts

The pipeline has five processing stages:

1. Inventory the workbook and source recordings.
2. Transcribe each configured recording.
3. Split recordings into deterministic base segments.
4. Transcribe the base segments independently.
5. Align candidate audio spans to script lines and generate the review package.

Finalization is intentionally separate from processing and reads only reviewed
selections.

Important project artifacts are:

- `project.json`: source locations, session mappings, and grouped settings.
- `source_lines.json`: normalized script-line records extracted from the
  workbook.
- `audio_inventory.json`: source hashes and probed media metadata.
- `transcripts/`: recording-level transcription caches.
- `normalized_sources/`: normalized PCM sources used for segmentation.
- `segments/`: base and derived candidate WAV files.
- `segments_manifest.json`: sample boundaries, audio metrics, transcripts, and
  relationships between base and derived segments.
- `segment_transcripts/`: base-clip and exact-candidate ASR caches.
- `alignment.json`: full alignment actions and diagnostics.
- `line_review.json`: the narrowed candidate lists, shared unmatched pool,
  statuses, and selected segment IDs used by the UI and finalizer.

Source workbooks and recordings are read-only. Generated paths live in the
project directory unless an explicit final output directory is supplied.

## Project configuration and caching

New projects serialize a complete grouped configuration. Alignment settings
must use the grouped schema; legacy flat alignment keys are rejected. Defaults
and validation are centralized in `dialogue_pipeline/alignment_settings.py`.

Expensive stages use source hashes, exact sample bounds, model identity, and
relevant settings as cache inputs. A batch-size change does not invalidate an
otherwise compatible transcription cache because it changes execution grouping,
not the requested decoding of an individual clip.

Reprocessing an existing destination preserves its configuration, session
mappings, caches, manual candidates, review selections, and retake marks. A
forced CLI metadata refresh writes `project.before-force.json` before replacing
project metadata.

Cancellation is cooperative. Inventory, ASR, segmentation, and alignment check
for cancellation at safe boundaries rather than interrupting file writes.

## Audio inventory, normalization, and refresh

The inventory records hashes, duration, sample rate, channels, bit depth, and
frame count. Sources not already in the export PCM shape are normalized once
into `normalized_sources/`. This includes 32-bit floating-point post-processed
WAV files.

Segmentation and derived cuts are sample-accurate and use the normalized source.
Configured fades are applied only when materializing the candidate WAV.

`refresh-audio` is a restricted fast path for post-processing that does not
change spoken content or timing. It re-probes configured recordings and
re-cuts every base, merged, bounded, trimmed, and manual segment at its stored
sample range. Source hashes, media metadata, audio metrics, and stored voice
bounds are updated; transcripts, alignment results, mappings, and selections
are retained. Refresh stops before replacement if sample rate, PCM shape, or
normalized frame count changed.

## Recording transcription

Recording transcription defaults to the `large-v3` Whisper model. Models are
stored in the shared user-level cache unless `DIALOGUE_VA_MODEL_CACHE` or
`transcription.model_cache` overrides it.

Recordings are processed one at a time because sessions have different script
hotwords and independent cache outputs. Within a recording, Whisper/VAD chunks
use batched inference.

`transcription.batch_size` accepts a positive integer or `"auto"`. Automatic
sizing estimates a conservative batch from free GPU memory after model loading.
`transcription.batch_size_max` caps the result and defaults to `32`. CPU mode or
unavailable GPU telemetry falls back to batch size `1`.

GPU execution uses the CUDA 12/cuDNN 9 runtime expected by the installed
Faster-Whisper/CTranslate2 stack. Automatic device selection falls back to CPU
INT8 if CUDA model initialization or inference fails.

## Segmentation

Segmentation creates deterministic base WAV files in `segments/` and records
them in `segments_manifest.json`.

The default acoustic boundary is at least 0.2 seconds below -40 dBFS. Adjacent
padded regions are clamped to one shared point inside the gap so the same audio
is not duplicated into two base clips.

When `segmentation.word_split_enabled` is true, recording-level Whisper word
timestamps add finer boundaries inside an acoustic region. This separates
adjacent lines or repeated takes whose pauses are shorter than the main silence
threshold. `word_split_max_boundaries` is a soft cap: exceptionally long pieces
continue to split at their strongest remaining word gaps until they satisfy
`word_split_max_segment_seconds`.

With `word_split_snap_enabled`, each ASR midpoint is moved to the quietest PCM
window within its word gap. Relevant settings are:

```json
{
  "segmentation": {
    "word_split_snap_enabled": true,
    "word_split_snap_search_seconds": 0.2,
    "word_split_snap_window_seconds": 0.02,
    "word_split_snap_max_rms_dbfs": -42.0,
    "voice_boundary_detection_enabled": true,
    "voice_boundary_vad_threshold": 0.5,
    "voice_boundary_breath_vad_threshold": 0.7
  }
}
```

Every padded base segment is analyzed with normal and strict Silero VAD
thresholds. Absolute `voice_bounds` are stored without destructively shortening
the base WAV. Alignment later decides whether and how to use those bounds.

## Independent segment transcription

Each base WAV is decoded independently, including voiced segments for which
the recording-level transcript is empty. Segment decoding uses
`vad_filter: false` and `condition_on_previous_text: false`; short lines are not
removed by VAD and context cannot leak between takes.

A nonempty result is discarded as a likely silence hallucination only when both
conditions are met:

- RMS is below `silence_rejection_max_rms_dbfs`.
- Whisper no-speech probability is above
  `silence_rejection_min_no_speech_probability`.

Independent clips are batched. `segment_transcription.batch_size` and
`segment_transcription.batch_size_max` follow the recording-transcription
rules. Clips longer than Whisper's 30-second batched window use full-audio
decoding to avoid truncation.

Unprompted clip ASR is canonical evidence. If it is empty, low-confidence, or
poorly ordered, the stage may store a script-prompted fallback. Prompted text
alone is never enough for `AUTO_OK`. Fallback clips are batched only when they
share the same prompt.

Base results are cached by source hash, exact bounds, model, and decoding
settings under `segment_transcripts/base/`. `align` ensures enabled segment
transcription is current, while `process` invokes it explicitly after
segmentation.

## Candidate discovery and global assignment

Every valid audio span is scored against every enabled line in its session.
Nonverbal scripts are excluded from text alignment. A global interval resolver
selects non-overlapping spans without requiring workbook row order. Repeated
nearby matches remain separate takes.

Competing line matches stay attached to spans during discovery and recovery.
`candidate_top_k` controls the number of internal alternatives.
`order_hint_weight` defaults to `0.0` and can be set to a small positive value
when recording order is useful only as a tie-breaker.

The candidate list in `line_review.json` is narrower than the diagnostic list
in `alignment.json`. Review pruning:

- removes lower-scoring spans contained by a reliable or structurally complete
  better candidate;
- retains candidates within 15 match-score points of the best;
- cuts earlier when an adjacent score gap is at least 12 points;
- preserves a useful strict fragment join, ahead of provisional joins, or a
  reliable candidate when ordinary pruning would remove all of them.

Manual selections are restored after regeneration even when the selected span
falls outside the new automatic cluster.

## Reliability and transcript fidelity

Candidate discovery is deliberately tolerant; `AUTO_OK` uses stricter,
separate gates. Diagnostics include:

- ordered similarity;
- fuzzy-token coverage and precision;
- anchored prefix and suffix coverage;
- leading and trailing token edits;
- extra-word count;
- duration plausibility;
- exact-span ASR status;
- technical score and clipping state.

Lines of three words or fewer default to complete coverage and precision plus
`short_line_min_ordered_score`. Longer lines use
`reliable_min_ordered_score`, `reliable_min_token_coverage`, and
`reliable_min_token_precision` but must also preserve their opening and closing
tokens.

Missing boundaries receive `MISSING_LINE_START`, `MISSING_LINE_END`, or
`MISSING_LINE_BOUNDARIES`. Extra speech receives `EXTRA_LINE_START` or
`EXTRA_LINE_END`. `reliable_max_boundary_missing_tokens` defaults to `0`.

Equivalent spoken forms are canonicalized before scoring, including common
contractions, `c'mon`/`come on`, and hesitation variants such as `err`, `erm`,
`uh`, and `um`.

Multi-sentence lines require every sentence-ending clause to reach
`reliable_min_clause_score`. Missing clauses receive `MISSING_SENTENCE`; clauses
in the wrong order receive `SENTENCE_ORDER_MISMATCH`. Ellipses around an
explicit hesitation are treated as pauses inside one clause.

Clipped candidates stay in review with `TECHNICAL_CLIPPING` by default. Within
the reliable cluster, the combined selection score can favor technical quality,
ASR confidence, and clause completeness over a trivial match-score advantage.

## Fragment recovery and boundary refinement

When a line is split, fragment recovery searches contiguous base-segment
windows, including neighbors of the globally selected span. It compares
independent base transcripts with recording-level word timestamps and keeps the
more complete ordered evidence.

`max_merge_segments` is the base-span limit for the primary aligner and
recovery. A legacy `fragment_join_max_segments` may raise but cannot lower that
limit. `fragment_join_max_actions` controls retained recovery alternatives, not
span size. `fragment_join_fallback_max_actions` separately bounds fallback
joins. `intra_segment_trim_max_actions_per_line` and
`intra_segment_trim_max_actions_per_segment` bound pause-based trim expansion.

Strict joins must pass whole-line similarity, token coverage, clause fidelity,
order, and precision. A provisional path admits spans that clearly improve an
incomplete fragment. A bounded number of promising alternatives are exact-WAV
transcribed before reliability is decided. Secondary matches can seed recovery;
`fragment_join_secondary_seed_min_match_score` defaults to `80`.
When a short boundary clause is audible but ASR cannot identify it reliably,
the span remains reviewable with `UNCERTAIN_BOUNDARY_AUDIO`.

Oversized base or merged segments can yield line-sized candidates from
independent word timestamps. Recovery also checks proper subspans of a selected
merge and restores a complete constituent span hidden by global assignment.

Pause-based edge joins use a 0.3-second gap by default and a bounded 0.1-second
fallback for otherwise complete joins. Trims entirely inside one base use a
0.4-second default. Boundaries are snapped to quiet PCM windows rather than raw
ASR timestamp midpoints. The untrimmed candidate remains available.

Repeated takes are blocked from `AUTO_OK` by token precision, detection of
voiced but untranscribed merged pieces (`MERGED_UNTRANSCRIBED_AUDIO`), and
`reliable_min_duration_plausibility` (`POSSIBLE_REPEATED_TAKES`). Repetition can
also supply a split point when Whisper stretches a boundary word across a pause;
both halves are then transcribed independently.

## Vocalizations and non-speech boundaries

Parenthesized directions and recognized sounds such as coughs, grunts, and
death rattles have line type `nonverbal`. They start in `REVIEW`, are never
auto-selected, and use the shared pool of audible segments not reliably matched
to normal dialogue. Unreliable normal-dialogue alternatives do not reserve their
constituent base segments from this pool.

Parenthesized performance cues and square-bracketed implementation details are
excluded from spoken-text matching. A parenthesized cue at the start or end of
a verbal line prevents `AUTO_OK` with
`EDGE_VOCALIZATION_UNVERIFIED`, because speech ASR cannot prove that the cue was
preserved. Pause-based trimming cannot remove that scripted edge. A nearby
standalone vocalization may be added as an extended review candidate.

Standalone paralinguistic boundary segments are treated as noise unless the
script includes the sound. Alignment retains the original merged span for
review and can create a clean alternative without the boundary segment.

For breath or room noise inside a textual segment, alignment composes the voice
bounds stored during segmentation and adds a short-padded candidate. Strict
breath VAD supplies only a proposed trailing endpoint: alignment scans forward
and creates the trim only if it finds a sufficiently quiet PCM window. This can
cut before a separated trailing breath while preventing VAD or imprecise ASR
timestamps from clipping a quiet release or unvoiced final consonant. When no
quiet separation exists, the tail is preserved. When reliable candidates are
otherwise tied, review selection prefers an intact segment over a generated
boundary trim.

## Exact-span candidate verification

With default verification settings, a candidate cannot become `AUTO_OK` until
the exact WAV that would be exported has an unprompted independent transcript.
Base candidates reuse their base-clip result; serious merged and derived spans
are decoded as continuous WAVs.

Uncached spans are deduplicated and batched. Cache identity includes the source
hash, exact sample bounds, fade/export settings, model, and decoding profile.
The exact transcript drives final similarity, clause, extra-word, and repetition
gates. Empty or failed verification receives `EXACT_SPAN_ASR_FAILED`.

Exact merged ASR is normally authoritative. One narrow boundary-consensus path
can restore an opening or closing hesitation omitted by merged ASR when the
ordered constituent transcript preserves it. Other short boundary clauses need
both constituent and recording-level support. Such candidates record
`boundary_clause_consensus: true` and use
`constituent_recording_boundary_consensus`.

## Manual copy/edit segments

Copy/edit materializes a `manual_edit` derived segment with a stable ID based on
the line, source candidate, and exact sample bounds. Saving is deliberately
untranscribed and records `manual_copy_edit_untranscribed` so the UI remains
responsive.

The manual-only Transcribe action runs exact-span candidate ASR on demand,
stores its cache under `segment_transcripts/candidates/`, and updates both the
manifest and review candidate with `candidate_asr_manual_copy_edit`. An empty
successful result is distinct from a segment that has never been transcribed.

The manual-only Delete action removes the manifest entry, review candidates,
generated WAV, and candidate transcript cache. It also clears matching selected
or suggested IDs, reranks remaining candidates, and restores an appropriate
`REVIEW` or `MISSING` status. The UI requires confirmation before deletion.

## Duplicate-line policies

`duplicate_line_policy` supports:

- `"review"`: keep identical script rows manual.
- `"weak_order"`: assign distinct chronological take groups to duplicate rows.
  If a nearby group collapses but enough distinct spans exist, distribute those
  spans chronologically. Only candidates meeting normal reliability anchor take
  groups; weaker matches attach without bridging genuine groups.
- `"reuse"`: permit one exact segment to satisfy duplicate text. Finalization
  also requires `export.allow_segment_reuse: true` or
  `--allow-segment-reuse`.

## Alignment configuration reference

The grouped configuration written for new projects is:

```json
{
  "alignment": {
    "span_search": {
      "max_segments": 8,
      "max_gap_seconds": 2.5,
      "max_duration_seconds": 35.0,
      "minimum_score": 45.0
    },
    "duplicates": {
      "policy": "weak_order",
      "take_group_gap_seconds": 12.0
    },
    "reliability": {
      "normal": {
        "minimum_score": 72.0,
        "minimum_margin": 8.0
      },
      "short": {
        "minimum_score": 88.0,
        "minimum_margin": 15.0
      }
    },
    "recovery": {
      "enabled": true,
      "max_candidates_per_line": 10,
      "fallback_candidates_per_line": 2,
      "max_segments": 8,
      "trim_oversized_segments": true,
      "trim_candidates_per_line": 2,
      "trim_minimum_gap_seconds": 0.4,
      "edge_trim_minimum_gap_seconds": 0.3,
      "edge_trim_fallback_minimum_gap_seconds": 0.1,
      "recover_complete_subspans": true,
      "clean_paralinguistic_boundaries": true,
      "boundary_cleanup_minimum_score": 85.0,
      "audio_boundaries": {
        "trim_non_speech_edges": true,
        "minimum_edge_seconds": 0.3,
        "pre_padding_seconds": 0.08,
        "post_padding_seconds": 0.12
      },
      "edge_cues": {
        "extend_adjacent_segments": true,
        "maximum_gap_seconds": 0.4,
        "maximum_segment_seconds": 5.0
      }
    },
    "quality": {
      "reject_clipping": true,
      "reject_untranscribed_merge": true,
      "review_edge_performance_cues": true,
      "boundary_clause_consensus": {
        "enabled": true,
        "minimum_exact_score": 85.0,
        "minimum_support_score": 95.0,
        "maximum_clause_words": 3
      }
    }
  }
}
```

The first grouped-settings version accidentally used 10-segment defaults;
migration restores the historical 8-segment limits.

## Review persistence

`line_review.json` contains every script line, line type, status, narrowed
candidates, selected and suggested segment IDs, and the audible unmatched pool.

Selecting a candidate changes status to `MANUALLY_REVIEWED`. Unselecting changes
it to `REVIEW`, or `MISSING` for a normal line without candidates. `RETAKE`
cannot contain a selected segment and survives later alignment runs.

Regeneration restores manual edit candidates and manual selections. For a
selected normal candidate, its segment and covered base segments are removed
from the unmatched pool. Selected unmatched segments are retained when needed
by nonverbal or missing lines.

## Finalization

The finalizer reads only `selected_segment_id` from `line_review.json`.
Unselected lines are omitted. Before copying it validates:

- target filenames;
- source WAV shape;
- segment reuse policy;
- target collisions;
- existing output files.

`--dry-run` performs validation without copying. `--allow-incomplete` exports
valid selections and writes failures to `finalization_errors.tsv`. `--overwrite`
permits intentional replacement. Segment reuse requires either project
configuration or `--allow-segment-reuse`.
