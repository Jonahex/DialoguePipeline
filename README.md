# Dialogue VA Pipeline

Local desktop and command-line tools for turning long voice-actor recordings
into reviewable per-take WAV files and then copying selected takes to the exact
filenames from an Excel script.

The pipeline never modifies the source workbook or source recordings.

## Setup

Requirements:

- Windows with Python 3.11 or newer.
- `ffmpeg`, `ffprobe`, and `ffplay` on `PATH`.
- An NVIDIA GPU is optional. CPU transcription is supported.

Create the project-local virtual environment:

```powershell
.\setup_venv.ps1
```

Check the installation:

```powershell
.\run_pipeline.ps1 doctor
```

## Desktop app

Launch the complete create/review/finalize workflow with:

```powershell
.\run_ui.ps1
```

`Open Project` accepts a project directory containing `project.json` and
`line_review.json`. `Create or Reprocess Project` first asks for a destination.
If it already contains `project.json`, the existing configuration, mappings,
caches, and manual selections are preserved. A scrollable settings dialog
shows every editable project setting in collapsible General, Transcription,
Segment Transcription, Segmentation, Alignment, and Export groups before the
cache-aware pipeline runs again. New destinations use the same settings dialog
after the UI asks for the lines workbook and recorded WAV directory, then run
inventory, transcription, segmentation, segment transcription, and alignment
in the background while showing the pipeline log.
The processing screen includes `Cancel`. Cancellation stops further work at the
next safe inventory, ASR, segmentation, or alignment boundary and returns to
the start screen without treating the cancellation as a pipeline failure.

The review screen lists script lines on the left and candidate segments on the
right. Filter lines by status, click a line-column header to sort, use the `▶`
cells to audition one WAV at a time, and select or unselect a take. Selections
are saved immediately to `line_review.json`. The selected line's context, full
text, and acting note are shown above the candidate table. Both tables have
permanently allocated vertical and horizontal scrollbars in addition to mouse
wheel scrolling. Nonverbal lines use the
name-sorted shared pool of audible segments that are not retained candidates
for any normal dialogue line. `Finalize Selected Lines` copies every selected
take to its target filename.

## Commands

### 1. Initialize a project

```powershell
.\run_pipeline.ps1 init `
  --workbook MaleElfYoung\ARG1RMElfYoung.xlsm `
  --audio-dir MaleElfYoung\Audio `
  --project-dir MaleElfYoung\PipelineWork
```

This creates `project.json`, `source_lines.json`, and `audio_inventory.json`.
Review the generated `sessions` section in `project.json` before transcription.
Each session specifies its source WAV and the script sheets or Excel rows that
may occur in that recording.

### 2. Transcribe

```powershell
.\run_pipeline.ps1 transcribe --project MaleElfYoung\PipelineWork
```

The default model is `large-v3`. Models are shared across projects in the
user-level cache shown by `doctor` (on Windows, normally
`%LOCALAPPDATA%\DialogueVAPipeline\models`). Set
`DIALOGUE_VA_MODEL_CACHE` to choose a different shared location, or set
`transcription.model_cache` in `project.json` for an explicit project override.
Recording transcription uses batched inference; configure
`transcription.batch_size` (default `"auto"`) to have the pipeline choose a
conservative batch from currently free GPU memory after the model is loaded.
Set a positive integer for an explicit override, and optionally lower
`transcription.batch_size_max` from its default of `32` to cap automatic
sizing. CPU execution and unavailable GPU telemetry fall back to batch size
`1`.
Use `--device cpu` when the CUDA 12/cuDNN 9 runtime is unavailable. For a quick
experiment, override the model:

```powershell
.\run_pipeline.ps1 transcribe `
  --project MaleElfYoung\PipelineWork `
  --session arg_maleelfyoung_gelarsathrin_row17pickup `
  --model small.en
```

### 3. Segment

```powershell
.\run_pipeline.ps1 segment --project MaleElfYoung\PipelineWork
```

This creates deterministic, sample-accurate temporary WAV files and
`segments_manifest.json`. Silence supplies conservative base boundaries;
alignment may create merged segment files when a line contains an internal
pause. Sources that are not already 48 kHz mono 16-bit PCM—including 32-bit
floating-point post-processed WAVs—are normalized once into the project's
`normalized_sources` cache before cutting.

When `segmentation.word_split_enabled` is true, Whisper word timestamps add up
to `word_split_max_boundaries` finer cuts inside an acoustic region. This
separates adjacent lines and repeated takes whose pauses are shorter than the
main silence threshold; alignment can still merge the pieces when they belong
to one line.

### 4. Transcribe the temporary segments

```powershell
.\run_pipeline.ps1 transcribe-segments --project MaleElfYoung\PipelineWork
```

Every base WAV is decoded independently, including voiced segments for which
the recording-level transcript was empty. Segment decoding always uses
`condition_on_previous_text: false` and `vad_filter: false`, so context from a
previous take cannot reorder a short phrase and Whisper VAD cannot discard
clips such as "Yes?". The segment files already contain the configured
pre/post padding from segmentation.

Independent clips are decoded in batches for substantially better accelerator
utilization. Configure the number of clips per inference batch with
`segment_transcription.batch_size`; the default is `"auto"` and uses the same
free-GPU-memory calculation. A positive integer is a fixed override, while
`segment_transcription.batch_size_max` caps automatic sizing (default `32`).

Unprompted clip ASR is the canonical evidence. When it is empty, low-confidence,
or has poor ordered similarity, a second script-prompted decode may be stored
as matching assistance. Prompted text alone is never sufficient for
`AUTO_OK`. Results are cached under `segment_transcripts` using the source
hash, exact sample bounds, model, and decoding settings. The Whisper model
itself still uses the shared user-level model cache.

This command is optional to type manually: `align` ensures that enabled segment
transcription is current, and `process` runs it explicitly after segmentation.
For a targeted retry, pass one or more exact IDs with
`--segment <segment_id>`; `--force` ignores a matching clip cache.

### 5. Align and generate the review package

```powershell
.\run_pipeline.ps1 align --project MaleElfYoung\PipelineWork
```

Outputs:

- `line_review.json`: every script line, its type and status, sorted candidate
  segments and transcripts, selected segment, plus the pool of audible
  unmatched segments used to review nonverbal and `MISSING` lines.
- `alignment.json`: machine-readable alignment details.

Use the desktop app to audition and change selections. Normal lines are
`AUTO_OK`, `REVIEW`, or `MISSING` after alignment. Selecting a candidate changes
the status to `MANUALLY_REVIEWED`; unselecting it changes the status to
`REVIEW`, or back to `MISSING` when no alignment candidate exists.

The review candidate list is intentionally narrower than the diagnostic
candidate set in `alignment.json`. It removes
lower-scoring spans contained by a better candidate, retains candidates within
15 score points of the best match, and cuts the list sooner when adjacent
scores have a gap of at least 12 points. Manual selections survive later
alignment runs even if the selected segment falls outside the new automatic
candidate cluster.

Every valid audio span is scored against every line enabled for the session; a
global interval resolver then chooses non-overlapping spans without requiring
the actor to follow sheet or row order. Repeated nearby matches to one line are
retained as separate takes. Competing line matches remain attached to each span
during discovery and fragment recovery instead of being expanded into
review-ineligible duplicate candidates. `candidate_top_k` controls that
internal alternative set. `order_hint_weight` is `0.0` by default and may be
set to a small positive value when recording order is useful as a tie-breaker.

Candidate discovery deliberately tolerates reordered or imperfect ASR text,
but `AUTO_OK` uses separate transcript-fidelity gates. Detailed diagnostics
remain in `alignment.json`, including ordered similarity, fuzzy-token coverage,
fuzzy-token precision, anchored prefix/suffix coverage, leading/trailing token
edits, extra-word count, and
exact-span ASR verification status. Lines of three words or fewer default to complete coverage
and precision plus `short_line_min_ordered_score`; longer lines use the more
tolerant `reliable_min_ordered_score`, `reliable_min_token_coverage`, and
`reliable_min_token_precision`, but must also preserve the beginning and end
of the script line. Missing boundaries receive `MISSING_LINE_START`,
`MISSING_LINE_END`, or `MISSING_LINE_BOUNDARIES`; words spoken before or after
the line receive `EXTRA_LINE_START` or `EXTRA_LINE_END`. By default,
`reliable_max_boundary_missing_tokens` is `0`, so even one omitted word in the
opening or closing comparison window prevents automatic acceptance. Common equivalent spoken
forms are canonicalized before scoring, including contractions such as
`could've`/`could have`, `I've`/`I have`, and colloquialisms such as
`c'mon`/`come on`. A merged span containing substantial,
non-quiet audio with no transcript is kept for review with reason
`MERGED_UNTRANSCRIBED_AUDIO` rather than accepted automatically. A take with
clipped samples also stays in review with `TECHNICAL_CLIPPING` by default.
Within the reliable candidate cluster, automatic selection uses the combined
selection score so technical quality, ASR confidence, and clause completeness
can favor a cleaner take over one with a trivially higher text score.

Multi-sentence lines also use clause-level completeness. Every clause separated
by `.`, `?`, or `!` must reach `reliable_min_clause_score`, otherwise the
candidate stays in review with reason `MISSING_SENTENCE`; clauses in the wrong
order receive `SENTENCE_ORDER_MISMATCH`. When a line is split, the fragment
joiner searches contiguous base-segment windows, including windows that merely
overlap or neighbor the global resolver's selection. This also covers a
single-clause line cut at a comma or another internal pause. The joiner compares
the independent base-segment transcripts with the session-level Whisper word
timestamps and uses the more complete, ordered evidence. By default it can
recover lines spread across as many as ten segments. Textless first or last
segments are rejected, so a duration hint cannot extend a candidate with an
empty segment. Strict joins must pass the configured whole-line similarity,
token coverage, clause fidelity, order, and precision thresholds. A second
provisional path admits spans that clearly improve the selected fragment but
whose preliminary fragment transcripts still miss a short clause. Those spans
are independently transcribed as exact WAVs before reliability is decided.
Up to two additional high-scoring provisional joins are retained as bounded
fallbacks even when the preliminary transcript does not improve every fidelity
dimension. This preserves promising alternate boundaries for exact-WAV
verification without restoring the former diagnostic candidate explosion.
Near-complete secondary line matches can also seed recovery; the default
`fragment_join_secondary_seed_min_match_score` is `80`. When a very short
opening or closing clause is audible but Whisper cannot identify it reliably,
the joined span remains available for playback with
`UNCERTAIN_BOUNDARY_AUDIO` instead of being silently pruned or auto-accepted.
If exact-span ASR shows that such a span belongs to a different script line than
the preliminary fragment text suggested, the candidate is reassigned to that
better line.
When one oversized base segment contains multiple performances, alignment can
also recover line-sized trimmed candidates from the independent base-ASR word
timestamps. Cuts are considered only at pauses of at least 0.4 seconds, must
produce materially more complete and precise text, and are independently
transcribed again before `AUTO_OK`. The untrimmed base candidate remains
available for comparison.
Inline performance cues such as `(laugh)` and `(hiccup)` are not treated as
spoken words during matching. Original fragments remain available.

`max_merge_segments` is the base-segment span limit for both the primary
aligner and fragment recovery. An existing `fragment_join_max_segments` value
may raise that recovery limit but can no longer lower it.
`fragment_join_max_actions` is different: it is the number of recovery
alternatives retained per line for exact-span verification, not a merge-size
limit. `fragment_join_fallback_max_actions` separately bounds fallback joins;
`intra_segment_trim_max_actions_per_line` bounds pause-based trims.

Repeated takes are blocked from `AUTO_OK` by three complementary checks:
repeated transcript words reduce token precision, merged spans with voiced but
untranscribed pieces receive `MERGED_UNTRANSCRIBED_AUDIO`, and candidates below
`reliable_min_duration_plausibility` receive `POSSIBLE_REPEATED_TAKES`. The
duration gate covers short segments where ASR collapses two or more audible
performances into a single exact transcript.

With the default segment-transcription and candidate-verification settings,
a candidate cannot become `AUTO_OK` until the exact WAV that would be exported
has an unprompted independent transcript. Base candidates reuse their
base-clip result; serious merged candidates are decoded again as one continuous
span and cached separately. Uncached merged spans are deduplicated and decoded
in batches using `segment_transcription.batch_size`; changing only batch size
does not invalidate recording, base-clip, or candidate-span transcript caches,
including caches written by older versions. The resulting exact-span text—not
the concatenated
base transcripts or script-prompted fallback—drives the final similarity,
clause, extra-word, and repetition gates. Empty or failed verification receives
`EXACT_SPAN_ASR_FAILED`.

Parenthesized directions and recognized vocalizations such as coughs, grunts,
and death rattles have line type `nonverbal`. They are excluded from normal
text matching, start in `REVIEW`, and are never auto-selected. The desktop app
shows all audible segments that were not reliable candidates for normal lines
when a nonverbal line is selected.

`duplicate_line_policy` supports:

- `"review"`: keep identical script text manual.
- `"weak_order"`: assign distinct chronological take groups to duplicate rows.
  If nearby takes collapse into one group but there are enough distinct spans,
  distribute those spans chronologically instead of leaving later rows missing.
- `"reuse"`: permit one exact segment to satisfy duplicate text. Pair this with
  `export.allow_segment_reuse: true` or `finalize --allow-segment-reuse`.

### Alignment configuration

New projects write a compact grouped configuration:

```json
{
  "alignment": {
    "span_search": {
      "max_segments": 10,
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
      "max_segments": 10,
      "trim_oversized_segments": true,
      "trim_candidates_per_line": 2,
      "trim_minimum_gap_seconds": 0.4
    },
    "quality": {
      "reject_clipping": true,
      "reject_untranscribed_merge": true
    }
  }
}
```

Existing projects using the original flat alignment keys remain supported.
When both forms specify the same option, the flat legacy value wins. Defaults
and validation are centralized in `dialogue_pipeline/alignment_settings.py`;
advanced recovery constants no longer need to be copied into every new
`project.json`.

### 6. Finalize selected files

Validate without copying:

```powershell
.\run_pipeline.ps1 finalize `
  --project MaleElfYoung\PipelineWork `
  --output MaleElfYoung\FinalWav `
  --dry-run
```

Copy selected files:

```powershell
.\run_pipeline.ps1 finalize `
  --project MaleElfYoung\PipelineWork `
  --output MaleElfYoung\FinalWav
```

To intentionally reuse one selected segment for multiple target filenames:

```powershell
.\run_pipeline.ps1 finalize `
  --project MaleElfYoung\PipelineWork `
  --output MaleElfYoung\FinalWav `
  --allow-segment-reuse
```

The finalizer reads only `selected_segment_id` from `line_review.json`.
Unselected lines are omitted. It validates target names, WAV format, segment
reuse, target collisions, and existing output files.

Use `--allow-incomplete` to export valid selections while retaining errors in
`finalization_errors.tsv`. Use `--overwrite` only when replacing an existing
final export intentionally.

## One-command processing

After reviewing `project.json`, run recording transcription, segmentation,
independent segment transcription, and alignment:

```powershell
.\run_pipeline.ps1 process --project MaleElfYoung\PipelineWork
```

All expensive stages are cached using source and settings hashes.
