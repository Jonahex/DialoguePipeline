# Dialogue VA Pipeline

Local command-line tools for turning long voice-actor recordings into reviewable
per-take WAV files and then copying selected takes to the exact filenames from an
Excel script.

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

### 4. Align and generate the review package

```powershell
.\run_pipeline.ps1 align --project MaleElfYoung\PipelineWork
```

Outputs:

- `A_line_review.xlsx`: every script line, linked top-three candidates,
  transcripts, linked suggested/selected takes, an editable `Selected Segment`,
  a formula-driven status pie chart below the line table, and an
  `Unmatched Segments` sheet with linked unmapped audio and script suggestions.
- `alignment.json`: machine-readable alignment details.

Click a candidate link to audition it. Copy an entire `Candidate 1`,
`Candidate 2`, or `Candidate 3` cell into `Lines!Selected Segment` to preserve
both its Segment ID and hyperlink. Segment IDs from the `Candidates` sheet or
the `Unmatched Segments` sheet also work.
Set `Status` to `SKIP` only when a line should intentionally be omitted.

Alignment defaults to `alignment.mode: "unordered"`. Every valid audio span is
scored against every line enabled for the session; a global interval resolver
then chooses non-overlapping spans without requiring the actor to follow sheet
or row order. Repeated nearby matches to one line are retained as separate
takes. `candidate_top_k` controls how many competing line matches are retained
per chosen span, while `order_hint_weight` is `0.0` by default and may be set to
a small positive value when recording order is known to be useful only as a
tie-breaker.

Set `alignment.mode` to `"sequence"` to use the legacy monotonic aligner.
`lookahead_lines`, `skip_line_penalty`, and `repeat_take_penalty` apply only to
that legacy mode.

Candidate discovery deliberately tolerates reordered or imperfect ASR text,
but `AUTO_OK` uses separate transcript-fidelity gates. The `Candidates` sheet
shows ordered similarity, fuzzy-token coverage, fuzzy-token precision, and
extra-word count. Lines of three words or fewer default to complete coverage
and precision plus `short_line_min_ordered_score`; longer lines use the more
tolerant `reliable_min_ordered_score`, `reliable_min_token_coverage`, and
`reliable_min_token_precision`. A merged span containing substantial,
non-quiet audio with no transcript is kept for review with reason
`MERGED_UNTRANSCRIBED_AUDIO` rather than accepted automatically.

Multi-sentence lines also use clause-level completeness. Every clause separated
by `.`, `?`, or `!` must reach `reliable_min_clause_score`, otherwise the
candidate stays in review with reason `MISSING_SENTENCE`. When successive
segments assigned to the same line represent successive clauses,
`fragment_join_enabled` creates an additional full-span candidate. The join is
accepted only when whole-line similarity, weakest-clause fidelity, and ordered
similarity improve by the configured `fragment_join_*` thresholds. Original
fragments remain in the candidate list, and the review workbook shows clause
and join diagnostics.

Repeated takes are blocked from `AUTO_OK` by three complementary checks:
repeated transcript words reduce token precision, merged spans with voiced but
untranscribed pieces receive `MERGED_UNTRANSCRIBED_AUDIO`, and candidates below
`reliable_min_duration_plausibility` receive `POSSIBLE_REPEATED_TAKES`. The
duration gate covers short segments where ASR collapses two or more audible
performances into a single exact transcript.

`local_asr_rescue.enabled` retries uncertain short base segments individually
using the strongest nearby script candidates as prompts. Accepted retries must
improve both script similarity and word confidence, are cached in
`segments_manifest.json`, and are identified in the `Candidates` sheet.

`nonverbal_policy` controls parenthesized directions and recognized
vocalizations such as coughs, grunts, and death rattles. They are always
excluded from normal text matching:

- `"review"` (default): create a `NONVERBAL_REVIEW` row with no suggested or
  selected segment. Audition the `Unmatched Segments` sheet and copy the
  intended Segment ID manually.
- `"skip"`: create the row with `SKIP` status and no candidate.
- `"weak_order"`: add manual-only phonetic/duration/order-hint candidates.
  These hints are intentionally never auto-selected and may be inaccurate.

The optional `vocalization_alignment` thresholds apply only when
`nonverbal_policy` is `"weak_order"`; its `enabled` switch can disable those
weak candidates without changing the line status.

`duplicate_line_policy` supports:

- `"review"`: keep identical script text manual.
- `"weak_order"`: assign distinct chronological take groups to duplicate rows
  only when enough groups exist.
- `"reuse"`: permit one exact segment to satisfy duplicate text. Pair this with
  `export.allow_segment_reuse: true` or `finalize --allow-segment-reuse`.

### 5. Finalize selected files

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

The finalizer reads only `Selected Segment`; `Suggested Best Segment` is
informational. It validates target names, WAV format, missing selections,
segment reuse, target collisions, and existing output files.

Use `--allow-incomplete` to export valid selections while retaining errors in
`finalization_errors.tsv`. Use `--overwrite` only when replacing an existing
final export intentionally.

## One-command processing

After reviewing `project.json`, run transcription, segmentation, and alignment:

```powershell
.\run_pipeline.ps1 process --project MaleElfYoung\PipelineWork
```

All expensive stages are cached using source and settings hashes.
