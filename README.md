# Dialogue VA Pipeline

Dialogue VA Pipeline is a local Windows application for turning long
voice-actor recordings into reviewable per-take WAV files and copying approved
takes to the exact filenames defined by an Excel dialogue script.

It provides a desktop workflow for creating a project, processing recordings,
reviewing candidate takes, making sample-accurate edits, requesting retakes,
and exporting final audio. The source workbook and source recordings are never
modified.

For pipeline internals, configuration details, cache behavior, and alignment
design, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Requirements

- Windows.
- Python 3.11 or newer.
- `ffmpeg`, `ffprobe`, and `ffplay` installed and available on `PATH`.
  **FFmpeg is required**: setup is not complete until all three commands are
  available.
- Enough disk space for normalized recordings, temporary segments,
  transcription caches, and final WAV files.
- An NVIDIA GPU is optional. **GPU transcription requires CUDA 12 and cuDNN 9**
  plus a compatible NVIDIA driver. Without that runtime, use CPU transcription.

The input workbook may be `.xlsx` or macro-enabled `.xlsm`. Recordings should
be WAV files; supported source formats are normalized by the tool before
segmentation.

## Setup

Create the project-local virtual environment:

```powershell
.\setup_venv.ps1
```

Verify Python packages and the required FFmpeg tools:

```powershell
.\run_pipeline.ps1 doctor
```

If `doctor` cannot find `ffmpeg`, `ffprobe`, or `ffplay`, add the FFmpeg `bin`
directory to `PATH` and run the check again.

## Desktop usage

The desktop app is the recommended way to use the pipeline:

```powershell
.\run_ui.ps1
```

### Create and process a project

1. Select **Create or Reprocess Project**.
2. Choose the directory in which the project should be stored.
3. For a new project, select the dialogue workbook and the directory containing
   the recorded WAV files.
4. Review the project settings. Settings are grouped into General,
   Transcription, Segment Transcription, Segmentation, Alignment, and Export.
5. Start processing. The app inventories the audio, transcribes it, creates
   candidate segments, aligns them to script lines, and opens the review screen.

Processing runs in the background and displays a log. **Cancel** stops at the
next safe processing boundary and returns to the start screen.

When processing a new project for the first time, review the generated session
mappings in `project.json`. Each session associates a source recording with the
workbook sheets or row ranges expected in that recording. Incorrect mappings
will produce poor candidate assignments.

### Open an existing project

Select **Open Project** and choose a directory containing `project.json` and
`line_review.json`. Existing settings, caches, manual candidates, selections,
and retake marks are retained when the project is reprocessed.

### Review candidates

The review screen shows script lines on the left and candidates for the
selected line on the right.

- Filter lines by status or click a line-column heading to sort.
- Use the play controls to audition a selected take or any candidate.
- Select or unselect candidates. Changes are saved immediately to
  `line_review.json`.
- Review the line context, full text, and acting note above the candidates.
- For nonverbal lines, choose from the shared pool of unmatched audible
  segments. Amber candidates are already selected for another line.
- Select **Mark for retake** to clear the current selection and assign `RETAKE`
  status.

Line statuses are:

- `AUTO_OK`: the pipeline selected a reliable take automatically.
- `REVIEW`: a person should choose or confirm a take.
- `MISSING`: no normal alignment candidate is available.
- `MANUALLY_REVIEWED`: a candidate was selected manually.
- `RETAKE`: the line should be recorded again.

### Copy and edit a candidate

Use **Copy/Edit** on a candidate to create a custom segment:

1. Adjust the start and end markers in the waveform view.
2. Use the mouse wheel over the waveform to zoom around the pointer.
3. Hold the right mouse button and drag to pan through the surrounding audio.
4. Preview the current range and save it.

Saving is sample-accurate and does not run transcription, so it completes
quickly. The new row is marked **Custom segment - not transcribed**.

Custom segments have two additional controls:

- **Transcribe** generates a transcript for that exact edited WAV. Afterward,
  the control becomes **Retranscribe**.
- **Delete** asks for confirmation, then removes the custom candidate, its WAV,
  and any transcript cache. If it was selected, the selection is cleared.

These controls are available only for segments created with **Copy/Edit**.

### Retakes and final export

- **Export retakes script** creates a workbook containing only lines marked
  `RETAKE`. It preserves the original workbook type, columns, and row
  formatting and omits sheets without retakes.
- **Finalize Selected Lines** copies selected takes to their target filenames.
  Unselected lines are omitted.

### Refresh after source-only audio changes

Use **Refresh Segments from Updated Audio** when completed recordings were
post-processed without changing spoken content or timing, for example after a
volume adjustment. The updated recordings must preserve their normalized
sample rate, channel/bit-depth shape, and frame count.

This operation updates the audio inventory and re-cuts existing segments while
preserving transcripts, mappings, candidates, manual edits, and selections. If
timing or spoken content changed, use **Create or Reprocess Project** instead.

## Command-line usage

The desktop app covers the complete workflow, but every processing stage is
also available from PowerShell.

### Initialize and process

```powershell
.\run_pipeline.ps1 init `
  --workbook MaleElfYoung\ARG1RMElfYoung.xlsm `
  --audio-dir MaleElfYoung\Audio `
  --project-dir MaleElfYoung\PipelineWork
```

Review the `sessions` section of the generated `project.json`, then run the
complete processing sequence:

```powershell
.\run_pipeline.ps1 process --project MaleElfYoung\PipelineWork
```

To force CPU transcription:

```powershell
.\run_pipeline.ps1 process `
  --project MaleElfYoung\PipelineWork `
  --device cpu
```

The stages can also be run separately:

```powershell
.\run_pipeline.ps1 transcribe --project MaleElfYoung\PipelineWork
.\run_pipeline.ps1 segment --project MaleElfYoung\PipelineWork
.\run_pipeline.ps1 transcribe-segments --project MaleElfYoung\PipelineWork
.\run_pipeline.ps1 align --project MaleElfYoung\PipelineWork
```

For targeted transcription work, `transcribe` accepts `--session`, while
`transcribe-segments` accepts one or more exact `--segment <segment_id>`
arguments. Both commands support model and device overrides; use `--force` to
ignore a matching transcription cache.

Use `--force` on an individual stage when its CLI help lists that option. Run
the following for complete command and option documentation:

```powershell
.\run_pipeline.ps1 --help
.\run_pipeline.ps1 process --help
```

### Refresh source audio

```powershell
.\run_pipeline.ps1 refresh-audio --project MaleElfYoung\PipelineWork
```

Use this only for source-only edits that preserve timing and normalized frame
count.

### Finalize selected files

Validate the export without copying:

```powershell
.\run_pipeline.ps1 finalize `
  --project MaleElfYoung\PipelineWork `
  --output MaleElfYoung\FinalWav `
  --dry-run
```

Copy selected takes:

```powershell
.\run_pipeline.ps1 finalize `
  --project MaleElfYoung\PipelineWork `
  --output MaleElfYoung\FinalWav
```

Useful finalization options:

- `--allow-incomplete`: export valid selections and write remaining errors to
  `finalization_errors.tsv`.
- `--overwrite`: intentionally replace existing output files.
- `--allow-segment-reuse`: allow one selected segment to satisfy multiple
  target filenames.

## Model cache

Transcription models are shared across projects. On Windows, the default cache
is normally `%LOCALAPPDATA%\DialogueVAPipeline\models`.

Set `DIALOGUE_VA_MODEL_CACHE` to choose another shared cache location, or set
`transcription.model_cache` in `project.json` for a project-specific override.
