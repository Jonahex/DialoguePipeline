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
`Refresh Segments from Updated Audio` is for source-only edits such as volume
adjustments. It re-probes the configured recordings and re-cuts every existing
base, merged, bounded, and trimmed segment at its stored sample range, then
opens the unchanged review package. It does not rerun transcription or
alignment. The updated recordings must preserve the spoken content, timing,
and normalized frame count.
Alignment settings must use the grouped schema documented below. A forced CLI
metadata refresh writes
`project.before-force.json` and carries the existing settings and reviewed
session mappings forward.
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
for any normal dialogue line. Each candidate also has a copy/edit control. It
opens a waveform view with 30 seconds of source context on either side of the
take and initially zooms so the segment fills most of the graph. Use the mouse
wheel over the waveform to zoom the time scale around the pointer, or hold the
right mouse button and drag to move left or right through the surrounding
audio. Drag the start and end markers, preview the current range, then save it
as a new manual candidate for the selected line. Saving writes a sample-accurate
WAV and records it in both `segments_manifest.json` and `line_review.json`;
canceling leaves the project unchanged. `Mark for retake` clears the line's
selected candidate and changes its status to `RETAKE`. `Export retakes script`
writes all lines with that status to their original worksheets, omitting sheets
without retakes while preserving the original script's workbook type, columns,
and row formatting. `Finalize Selected Lines` copies every selected take to its
target filename.

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

### Refresh segments after source-only audio edits

If completed source recordings were post-processed without changing their
spoken content or timing, refresh the project without ASR or alignment:

```powershell
.\run_pipeline.ps1 refresh-audio --project MaleElfYoung\PipelineWork
```

This updates `audio_inventory.json`, including source hashes and media
metadata, and overwrites all WAVs represented in `segments_manifest.json`
using their existing sample boundaries. Segment source hashes, audio metrics,
and stored voice bounds are updated while transcripts, candidate mappings,
alignment results, and manual selections remain unchanged. The command stops
before replacing segment files if an updated source normalizes to a different
sample rate, channel/bit-depth shape, or frame count.

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
`1`. The batch is made from the recording's Whisper/VAD chunks; configured
recordings are still processed one at a time because each has different
hotwords and cache output.
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

The defaults treat post-processed audio below -40 dBFS for at least 0.2 seconds
as a take boundary. Adjacent padded regions are clamped to a shared point in
that gap, so a take is never duplicated into both base clips.

When `segmentation.word_split_enabled` is true, Whisper word timestamps add up
to `word_split_max_boundaries` finer cuts inside an acoustic region. This
separates adjacent lines and repeated takes whose pauses are shorter than the
main silence threshold; alignment can still merge the pieces when they belong
to one line. The boundary count is a soft cap: exceptionally long pieces add
their strongest remaining word gaps until they are no longer than
`word_split_max_segment_seconds`. Each ASR midpoint is snapped to the quietest
nearby PCM window inside its word gap before the base clips are written.

Segmentation also analyzes every padded base segment with normal and strict
Silero VAD thresholds and stores the absolute results in
`segments_manifest.json` as `voice_bounds`. It does not destructively shorten
the base WAV: alignment later decides whether an edge sound belongs to the
script and composes the stored bounds across merged candidates. The defaults
are:

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

### 4. Transcribe the temporary segments

```powershell
.\run_pipeline.ps1 transcribe-segments --project MaleElfYoung\PipelineWork
```

Every base WAV is decoded independently, including voiced segments for which
the recording-level transcript was empty. Segment decoding always uses
`vad_filter: false`, but a nonempty result is discarded when both its RMS is
below `silence_rejection_max_rms_dbfs` and Whisper's no-speech probability is
above `silence_rejection_min_no_speech_probability`. This prevents silent
clips from becoming candidates such as `Thank you.`. It also uses
`condition_on_previous_text: false`, so context from a previous take cannot
reorder a short phrase. The segment files already contain the configured
pre/post padding from segmentation.

Independent clips are decoded in batches for substantially better accelerator
utilization. Configure the number of clips per inference batch with
`segment_transcription.batch_size`; the default is `"auto"` and uses the same
free-GPU-memory calculation. A positive integer is a fixed override, while
`segment_transcription.batch_size_max` caps automatic sizing (default `32`).
The value is a maximum, so a final partial batch is smaller. Prompted fallback
clips are batched when they share the same script prompt; clips with different
prompts require separate inference calls. A clip longer than Whisper's
30-second batched window is decoded through the full-audio path instead of
being truncated. `segments_manifest.json` records the primary and prompted
inference-batch counts for the latest run.

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
`REVIEW`, or back to `MISSING` when no alignment candidate exists. A line
marked `RETAKE` cannot have a selected candidate, and that mark survives later
alignment runs.

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
`c'mon`/`come on`. Hesitation variants such as `err`, `erm`, `uh`, and `um`
are also equivalent. A merged span containing substantial, non-quiet audio with
no transcript is kept for review with reason `MERGED_UNTRANSCRIBED_AUDIO`
rather than accepted automatically. Quiet padding rejected by ASR is ignored,
and an exact-span transcript whose first or last word is far from the audio
edge is rejected only when the constituent transcripts also show that a
repeated take was collapsed. A take with clipped samples also stays in review
with `TECHNICAL_CLIPPING` by default.
Within the reliable candidate cluster, automatic selection uses the combined
selection score so technical quality, ASR confidence, and clause completeness
can favor a cleaner take over one with a trivially higher text score.

Multi-sentence lines also use clause-level completeness. Every clause separated
by a sentence-ending `.`, `?`, or `!` must reach
`reliable_min_clause_score`. Ellipses around an explicit hesitation such as
`I... err... I misspoke` are treated as pauses inside one clause. Otherwise the
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
When one oversized base segment contains multiple performances, including a
base segment already covered by a merged primary action, alignment can recover
line-sized trimmed candidates from the independent base-ASR word timestamps.
It also checks proper subspans of a selected merged action and restores any
constituent span that independently contains the complete script line. This
prevents the interval resolver from hiding a clean base candidate merely
because a longer overlapping merge won the primary path.
Fragment joins can also trim the first or last base segment at the same
pause-based boundaries, allowing a line to cross a shared take-boundary
segment. Edge joins use a 0.3-second pause by default and a bounded 0.1-second
fallback for otherwise text-complete joins; trims wholly inside one base retain
the more conservative 0.4-second default. All such candidates are independently
transcribed before they can become `AUTO_OK`. When two
complete copies of the same line are adjacent but Whisper stretches a boundary
word across the pause, the repetition itself supplies a split boundary and
both halves are independently transcribed again before `AUTO_OK`. The
untrimmed candidates remain available for comparison.
Inline performance cues such as `(laugh)` and `(hiccup)` are not treated as
spoken words during text matching. A cue at the beginning or end of a spoken
line nevertheless prevents `AUTO_OK` with
`EDGE_VOCALIZATION_UNVERIFIED`, because speech ASR cannot prove that the
performance was preserved. Pause-based trimming is not allowed to remove that
scripted edge, and a nearby standalone vocalization segment is added as an
extended review candidate when its transcript resembles laughter, a sigh,
breathing, or another common vocalization. Original fragments remain
available.

`max_merge_segments` is the base-segment span limit for both the primary
aligner and fragment recovery. An existing `fragment_join_max_segments` value
may raise that recovery limit but can no longer lower it.
`fragment_join_max_actions` is different: it is the number of recovery
alternatives retained per line for exact-span verification, not a merge-size
limit. `fragment_join_fallback_max_actions` separately bounds fallback joins;
`intra_segment_trim_max_actions_per_line` and
`intra_segment_trim_max_actions_per_segment` bound pause-based trims.

Repeated takes are blocked from `AUTO_OK` by three complementary checks:
repeated transcript words reduce token precision, merged spans with voiced but
untranscribed pieces receive `MERGED_UNTRANSCRIBED_AUDIO`, and candidates below
`reliable_min_duration_plausibility` receive `POSSIBLE_REPEATED_TAKES`. The
duration gate covers short segments where ASR collapses two or more audible
performances into a single exact transcript.

Standalone boundary segments transcribed as common paralinguistic sounds such
as `Pfft.`, laughter, sighs, coughs, or breathing are treated as noise unless
the script includes the sound at that edge. Alignment retains the original
merged span for review, rejects it from `AUTO_OK` when exact-span ASR omits the
sound, and creates a clean candidate without that boundary segment. This
handles imprecise segmentation without blindly dropping a whole textual base
segment. For extra breath or room noise inside a textual segment, alignment
uses the voice bounds computed during segmentation and creates an additional
candidate with short pre/post padding. It composes those bounds across merged
base segments; only a newly created intra-segment boundary or a legacy
manifest without stored bounds requires live analysis. Scripted
edge-performance cues suppress the corresponding trim. The
untrimmed candidate remains available for review and cannot become `AUTO_OK`
while its cleaned alternative exists. The strict bounds detect trailing
breaths that the normal speech threshold accepts as voice; the final ASR word
timestamp is a hard lower bound, so this pass cannot cut recognized dialogue.

Pause-based cuts inside a base segment use ASR word gaps to locate candidate
boundaries. The midpoint is only the initial estimate: alignment snaps it to
the quietest nearby PCM window inside the word gap. This avoids splitting a
word-ending release consonant that extends beyond Whisper's timestamp.

Exact merged-span ASR normally remains authoritative. One narrow exception is
an opening or closing hesitation that merged ASR drops while the complete,
ordered constituent transcription retains it. Other short boundary clauses
still require both the constituent base transcription and the recording-level
transcription to support the scripted boundary while the rest of the exact
span is complete. Such candidates use
`constituent_recording_boundary_consensus` and record
`boundary_clause_consensus: true` in `alignment.json`.

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
  Only candidates meeting the normal reliability score anchor take groups;
  weaker fuzzy matches attach to the nearest anchor without bridging or
  rebalancing genuine groups.
- `"reuse"`: permit one exact segment to satisfy duplicate text. Pair this with
  `export.allow_segment_reuse: true` or `finalize --allow-segment-reuse`.

### Alignment configuration

New projects write a complete grouped configuration. Every effective alignment
setting is serialized and available in the settings window:

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

Flat alignment keys are rejected with a configuration error; only the grouped
schema is accepted. Defaults and validation are centralized in
`dialogue_pipeline/alignment_settings.py`.
The first grouped-settings version's accidental 10-segment defaults migrate
back to the historical 8-segment limits.

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
Consequently, changing only a batch size and rerunning without the stage's
force option tests cache reuse rather than inference speed. Larger batches are
also not guaranteed to be faster: feature extraction remains per clip,
autoregressive decoding time depends on the longest item in a batch, and very
large batches can reduce throughput on some GPUs. Compare several fixed sizes
on the same uncached inputs when tuning for a particular machine.
