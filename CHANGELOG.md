# Changelog

## 0.5.4 Early Access (2026-11-05)

- A small set of base presets now ships inside the add-on, so there is something to start from
  without downloading anything.
- New *Copy Diagnostics* and *Report a Bug* buttons: copy your version, GPU and settings to the
  clipboard and open the bug-report form.

## 0.5.3 Early Access (2026-11-03)

- New liquid setting *Sampling* (spray, foam and bubbles; High Quality): *Multiple* tries
  eight spots per cell, so particles follow the splash smoothly instead of in cell-sized
  blocks. The sampling rates are scaled down automatically, so the particle count stays about
  the same as with Single. *Single* matches Blender; Classic always uses it.
- New liquid setting *Thin Walls* (spray, foam and bubbles; High Quality): checks the path of
  each particle so fast ones no longer jump through thin obstacles. *Auto* checks in steps of
  half a cell (only fast particles cost extra), *Fixed Steps* matches Mantaflow, *Off* matches
  Blender. Classic quality always uses Off.
- New preference *Device*: *Auto* (default), *GPU* or *CPU*. Auto bakes on the graphics
  card; when the bake doesn't fit in free graphics memory it keeps the card and pages the
  grids through system RAM (about 2x slower, still 30x+ faster than the CPU; Linux for now),
  and uses the CPU only without a usable card (or on Windows), instead of refusing the bake.
  The domain panel shows which one Auto will use. The command line takes
  `--device auto|cpu|cuda:0`; the bake log and status.json record the mode.

## 0.5.2 Early Access (2026-11-03)

Quality modes and a better liquid surface.

- New domain setting *Quality*: *Classic* (Blender's method, unchanged) or *High Quality*
  (default for new domains). Smoke: advection-reflection, more swirl and fine detail
  (about 1.7x time per step, about 10% more memory). Liquid: ghost-fluid free surface,
  a more coherent splash sheet and fewer stray drops.
- Liquid *Surface Tension* (N/m, water 0.072; High Quality only). Rounds drops and calms
  pools; bakes are slower because the time step is capped.
- Liquid Mesh Generator *High Quality* (default): anisotropic particle kernels (Yu and
  Turk), flatter surfaces and thin sheets; a still tank meshes at its true depth.
  *Final* and *Fast* remain.
- Licence activation: enter your key and email, accept the Licence and Terms of Service (links and a checkbox in Preferences) and press Activate. Presets download and support requests need an activated key. A key allows 5 new computers in any 12 months; only a one-way hash of your system ID is sent.
- Bake file schema 14. Older bakes and command-line runs keep Classic and Final.
- Combustion fire model and the Campfire and Explosion presets (preset pack 0.2.0).

## 0.5.1 Early Access (2026-11-01)

Windows fixes, update button, clearer tooltips.

- VDB `velocity` is now cell-centred metres per second (was raw staggered solver units), so Blender volume motion blur works; README documents it and the normalised `temperature`.
- Windows: the bake no longer fails when Blender has a file open. Replacing `status.json`,
  the VDB frames and the USD file waits and retries for up to 8 seconds instead of raising
  a permission error. `status.json` is informational and never stops a bake.
- Windows: the "is another bake running?" check no longer terminates the process that owns
  a stale `bake.lock` (`os.kill(pid, 0)` kills on Windows).
- Detect GPU without Warp installed now says so and offers *Install GPU Engine* instead of
  showing a Python error.
- Support: tickets include the last 20 Blender warnings and errors (checkbox, on by default),
  shown in the preview first.
- Cache and Cache Root fields accept `//` blend-relative paths without Blender's warning.
- Liquid: the Volume holds its first and last frame outside the baked range (no more
  "Cannot find 'phi' grid"), is hidden with the eye icon instead of the greyed-out monitor
  icon, and the Volume to Mesh link is repaired if the Volume was deleted and recreated.
- New: *Refresh VDB on Bake* reloads the baked files when a bake ends.
- Faster Bake/Resume start on heavy scenes: the result objects are left out of the many
  frame evaluations the export does (a few seconds saved on a 250-frame liquid scene).
- New: Preferences > Updates (check, download with checksum, install; free during early access).
- Every setting, dropdown option and button has a one-sentence plain-language tooltip.

## 0.5.0 Early Access (2026-10-30)

Secondary particles for liquids: spray, foam and bubbles.

- Solver: Mantaflow's secondary-particle step (wave crest, trapped air and energy
  potentials, sampling, life, bubble buoyancy and drag), once per frame like Blender.
  Matches a Blender bake exactly when replayed on Blender's own grids (counts, types,
  positions, velocities, life, all 60 frames of the dam scene). Optional finer
  particle grid (Upres 1-4) and Delete / Push Out at obstacles.
- Output: one time-sampled USD file (`vdb/particles_snd.usdc`) with a Points prim per
  type, kept as per-frame files while baking so a bake can resume. The add-on's
  Import Particles button loads them as animated point clouds with a material per type.
  Needs Blender's bundled Python (it has `pxr`).
- Add-on: Secondary Particles panel with Blender's names and defaults; the VRAM
  estimate includes the particle grids and a particle budget (rough).
- Viscosity: Blender's implicit viscosity solve (Strength, on the liquid panel), matching a
  Blender bake of a viscous dam (volume and centre of mass exact at Strength 0.05).
  About 16 ms/frame extra at 64 cells, 0.5 s at 128 cells.
- Diffusion: Blender'"'"'s Diffusion panel (Base, Exponent) as an implicit velocity diffusion before
  the viscosity solve. Matches a Blender bake of a dam (volume exact, centre of mass within
  3 mm over 60 frames). Surface tension is left out: Blender'"'"'s liquid script never applies it.
- Fixed: the volume correction measured the liquid volume including the border layer, which
  drained real fluid when the liquid spread over a wall or floor. This also changes
  0.4.1 liquids that touch walls (they now keep more volume).
- Counts are about half of Blender's with volume correction on, because a correct volume
  has less deep fluid (fewer bubbles). Raise the sampling rates or switch the correction off to match.

## 0.4.1-alpha (2026-10-30)

Liquid surface and volume.

- Surface now follows Blender's mesh step: the `phi` VDB is resampled to
  `mesh_scale` (default 2) and built with the improved particle level set
  (Blender's "Final" generator) or the faster union generator, with Blender's
  radius, smoothing and concavity settings.
- Volume correction (on by default; switch it off in the Liquid panel to match Blender's volume gain): removes the volume gain that FLIP
  accumulates, by adding a small target divergence in the pressure solve.
  Tracks inflows and open-side outflow.
- Add-on: mesh and volume-correction settings, VRAM estimate includes the mesh grid.
- Changing mesh settings marks a bake as stale.

## 0.4.0-alpha (2026-10-30)

Adds liquid (FLIP), following Blender's Mantaflow liquid.

- Solver: FLIP particles with a level-set surface, liquid flows (geometry,
  inflow, outflow), static and moving obstacles, open domain sides.
- Output: level-set `phi` VDB and a particle PLY; the add-on imports a Volume to
  Mesh setup.
- Improvements, on by default and switchable off to match Blender exactly:
  Floor Fix (particles near walls no longer gain speed), Robust Collisions
  (particles and thin walls no longer leak through obstacles).
- Add-on: liquid domain type, liquid panels, VRAM estimate.
- Not yet: volume correction, better surface, in-solver meshing (0.4.1).

## 0.3.0-alpha (2026-10-30)

Adds noise (Blender's wavelet turbulence, "High Resolution").

- Solver: a one-way post-pass builds a finer density (and flame) grid, `upres`
  times finer per axis, from the low-res state. The low-res sim is unchanged.
  The tile and the pass match Blender's noise reference scene.
- Output: the VDB `density`/`flame` grids are high-res, so the existing
  materials work as they are.
- Add-on: Domain > Noise panel (enable, upres, strength, scale, time); the
  VRAM estimate and the limit check include the noise grids.

## 0.2.0-alpha (2026-10-30)

Adds fire and moving obstacles. Noise/upres later.

- Packaging: one ~70 KB zip for all platforms. Warp is downloaded on first use
  with an **Install GPU Engine** button (was a 135 MB zip per platform).
- Solver: moving (animated) obstacles push the smoke, with Blender's obstacle
  velocity (vertex motion per substep, extrapolated and applied at the walls).
  Matches Blender's moving-obstacle reference scene to ~0.1%.
- Add-on: Flow type (Smoke / Fire / Fire + Smoke) with Flame Rate, a per-flow
  emit frame range (a short range makes an explosion), a domain Fire panel,
  animated obstacles, and VRAM estimates that count fire and moving obstacles.

## 0.1.0-alpha

First release. Smoke only.

- GPU smoke solver on NVIDIA Warp, following Blender's Mantaflow gas step order:
  MacCormack advection, buoyancy, vorticity confinement, MGPCG pressure solve.
- Flows and obstacles from mesh objects; border collisions; smoke dissolve.
- OpenVDB sequence output, loaded as a Volume object.
- Blender add-on: Domain, Flow and Obstacle panels, VRAM estimate, Bake,
  Resume, Cancel and Free Bake, progress and ETA.
- Command-line bake (`openbillow_bake.py`) for the render farm, with checkpoints
  and `--resume`.
- Packaged as a Blender extension (Blender 4.5+), zips for linux-x64 and
  windows-x64 (the Windows zip is built but untested on Windows).
