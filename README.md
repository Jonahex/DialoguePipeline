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
`line_review.json`. `Create New Project` asks for the lines workbook, recorded
WAV directory, and a destination project directory, then runs inventory,
transcription, segmentation, segment transcription, and alignment in the
background while showing the pipeline log.

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
`transcription.batch_size` (default `16`) according to available GPU memory.
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
`segment_transcription.batch_size`; the default is `16`. Reduce it if GPU
memory is insufficient.

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
candidate set in `alignment.json`. It keeps primary line matches, removes
lower-scoring spans contained by a better candidate, retains candidates within
15 score points of the best match, and cuts the list sooner when adjacent
scores have a gap of at least 12 points. Manual selections survive later
alignment runs even if the selected segment falls outside the new automatic
candidate cluster.

Every valid audio span is scored against every line enabled for the session; a
global interval resolver then chooses non-overlapping spans without requiring
the actor to follow sheet or row order. Repeated nearby matches to one line are
retained as separate takes. `candidate_top_k` controls how many competing line
matches are retained per chosen span, while `order_hint_weight` is `0.0` by
default and may be set to a small positive value when recording order is known
to be useful only as a tie-breaker.

Candidate discovery deliberately tolerates reordered or imperfect ASR text,
but `AUTO_OK` uses separate transcript-fidelity gates. Detailed diagnostics
remain in `alignment.json`, including ordered similarity, fuzzy-token coverage,
fuzzy-token precision, extra-word count, and exact-span ASR verification
status. Lines of three words or fewer default to complete coverage
and precision plus `short_line_min_ordered_score`; longer lines use the more
tolerant `reliable_min_ordered_score`, `reliable_min_token_coverage`, and
`reliable_min_token_precision`. A merged span containing substantial,
non-quiet audio with no transcript is kept for review with reason
`MERGED_UNTRANSCRIBED_AUDIO` rather than accepted automatically.

Multi-sentence lines also use clause-level completeness. Every clause separated
by `.`, `?`, or `!` must reach `reliable_min_clause_score`, otherwise the
candidate stays in review with reason `MISSING_SENTENCE`; clauses in the wrong
order receive `SENTENCE_ORDER_MISMATCH`. When a line is split, the fragment
joiner searches contiguous base-segment windows, including windows that merely
overlap or neighbor the global resolver's selection. This also covers a
single-clause line cut at a comma or another internal pause. The joiner compares
the independent base-segment transcripts with the session-level Whisper word
timestamps and uses the more complete, ordered evidence. By default it can
recover lines spread across as many as six segments. Textless first or last
segments are rejected, so a duration hint cannot extend a candidate with an
empty segment. Strict joins must pass the configured whole-line similarity,
token coverage, clause fidelity, order, and precision thresholds. A second
provisional path admits spans that clearly improve the selected fragment but
whose preliminary fragment transcripts still miss a short clause. Those spans
are independently transcribed as exact WAVs before reliability is decided.
Inline performance cues such as `(laugh)` and `(hiccup)` are not treated as
spoken words during matching. Original fragments remain available.

`max_merge_segments` is the base-segment span limit for both the primary
aligner and fragment recovery. An existing `fragment_join_max_segments` value
may raise that recovery limit but can no longer lower it.
`fragment_join_max_actions` is different: it is the number of recovery
alternatives retained per line for exact-span verification, not a merge-size
limit.

Repeated takes are blocked from `AUTO_OK` by three complementary checks:
repeated transcript words reduce token precision, merged spans with voiced but
untranscribed pieces receive `MERGED_UNTRANSCRIBED_AUDIO`, and candidates below
`reliable_min_duration_plausibility` receive `POSSIBLE_REPEATED_TAKES`. The
duration gate covers short segments where ASR collapses two or more audible
performances into a single exact transcript.

Before a candidate can become `AUTO_OK`, the exact WAV that would be exported
must have an unprompted independent transcript. Base candidates reuse their
base-clip result; serious merged candidates are decoded again as one continuous
span and cached separately. Uncached merged spans are deduplicated and decoded
in batches using `segment_transcription.batch_size`; changing only batch size
does not invalidate their transcript cache. The resulting exact-span text—not the concatenated
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
- `"weak_order"`: assign distinct chronological take groups to duplicate rows
  only when enough groups exist.
- `"reuse"`: permit one exact segment to satisfy duplicate text. Pair this with
  `export.allow_segment_reuse: true` or `finalize --allow-segment-reuse`.

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
