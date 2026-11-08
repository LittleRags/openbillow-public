# SPDX-License-Identifier: GPL-3.0-or-later
"""Preferences and the Object.openbillow property groups. Field names and defaults
copy Blender's FluidDomain/Flow/Effector settings, so export.py's `*_kw` helpers
read them unchanged."""
import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy.types import AddonPreferences, PropertyGroup


class OpenBillowWeights(PropertyGroup):
    gravity: FloatProperty(name="Gravity", default=1.0, min=0.0, soft_max=10.0,
                           description="How strong gravity is for this fluid. 1 is normal, 0 is weightless.")
    all: FloatProperty(name="All", default=1.0, soft_min=0.0, soft_max=1.0,
                       description="One slider that makes every force field in the scene stronger or weaker.")
    force: FloatProperty(name="Force", default=1.0, soft_min=0.0, soft_max=1.0,
                         description="Strength of ordinary Force fields (push or pull). 0 switches them off for this fluid.")
    wind: FloatProperty(name="Wind", default=1.0, soft_min=0.0, soft_max=1.0,
                        description="Strength of Wind fields. 0 switches them off for this fluid.")
    vortex: FloatProperty(name="Vortex", default=1.0, soft_min=0.0, soft_max=1.0,
                          description="Strength of Vortex fields (swirling). 0 switches them off for this fluid.")
    magnetic: FloatProperty(name="Magnetic", default=1.0, soft_min=0.0, soft_max=1.0,
                            description="Strength of Magnetic fields. 0 switches them off for this fluid.")
    harmonic: FloatProperty(name="Harmonic", default=1.0, soft_min=0.0, soft_max=1.0,
                            description="Strength of Harmonic fields (spring-like pull towards a point). 0 switches them off for this fluid.")
    charge: FloatProperty(name="Charge", default=1.0, soft_min=0.0, soft_max=1.0,
                          description="Strength of Charge fields (attract or repel). 0 switches them off for this fluid.")
    drag: FloatProperty(name="Drag", default=1.0, soft_min=0.0, soft_max=1.0,
                        description="Strength of Drag fields (slow the fluid down). 0 switches them off for this fluid.")
    turbulence: FloatProperty(name="Turbulence", default=1.0, soft_min=0.0, soft_max=1.0,
                              description="Strength of Turbulence fields (random swirling). 0 switches them off for this fluid.")


def _quality_updated(self, context):
    """Tie the liquid surface builder to Quality; the user can still change it afterwards."""
    self.mesh_generator = "ANISOTROPIC" if self.quality == "HIGH" else "IMPROVED"


class OpenBillowDomain(PropertyGroup):
    domain_type: EnumProperty(name="Domain Type", default="GAS", items=[
        ("GAS", "Gas", "Smoke, fire and explosions."), ("LIQUID", "Liquid", "Water, honey and other liquids.")], description="Smoke and fire, or liquid like water and honey.")
    use_apic: EnumProperty(name="Method", default="FLIP", items=[
        ("FLIP", "FLIP", "Lively and splashy, but can look a little noisy."), ("APIC", "APIC", "Smoother, and keeps swirling motion better. Slightly less splashy.")], description="How the liquid is moved around. FLIP is livelier and splashier; APIC is smoother and keeps swirls better.")
    flip_ratio: FloatProperty(name="FLIP Ratio", default=0.97, min=0.0, max=1.0,
                              description="Calm and smooth (0) versus lively and splashy (1). Around 0.97 suits most liquids.")
    particle_number: IntProperty(name="Particles/Cell", default=2, min=1, max=5,
                                 description="How many particles fill each grid cell along each direction (2 gives 8 per cell). More is smoother but slower.")
    particle_minimum: IntProperty(name="Minimum", default=8, min=0, max=1000,
                                  description="If a cell has fewer particles than this, new ones are added so there are no holes.")
    particle_maximum: IntProperty(name="Maximum", default=16, min=0, max=1000,
                                  description="If a cell has more particles than this, the extras are removed to save memory.")
    particle_radius: FloatProperty(name="Radius", default=1.0, min=0.0, max=10.0,
                                   description="Size of each particle, in grid cells. Bigger particles give a thicker, smoother liquid skin.")
    particle_band_width: FloatProperty(name="Narrow Band", default=3.0, min=0.0, max=1000.0,
                                       description="How many cells deep from the surface the liquid is tracked in fine detail. Raise it if the surface looks ragged.")
    particle_randomness: FloatProperty(name="Randomness", default=0.1, min=0.0, max=10.0,
                                       description="Shakes the particles slightly off their starting grid so the liquid does not show a regular pattern.")
    use_delete_in_obstacle: BoolProperty(name="Delete In Obstacle", default=False,
                                         description="Remove any liquid that ends up inside an obstacle, so it cannot get stuck there.")
    flip_wall_fix: BoolProperty(name="Floor Fix", default=True,
                                description="Calms liquid that is too lively right next to the floor and walls.")
    robust_collisions: BoolProperty(name="Robust Collisions", default=True,
                                    description="Stops liquid leaking through thin walls and obstacles. Slightly different from Blender's own result.")
    collision_margin: FloatProperty(name="Collision Margin", default=0.1, min=0.0, max=1.0,
                                    description="How far, in grid cells, liquid is kept away from obstacles.")
    volume_correction: BoolProperty(name="Volume Correction", default=True,
                                    description="Stops the liquid slowly shrinking or growing, so the amount stays what you put in.")
    volume_correction_rate: FloatProperty(name="Correction Rate", default=0.5, min=0.0, max=2.0,
                                          description="How quickly a gain or loss of liquid is corrected each frame. 0.5 corrects half of it.")
    use_fractions: BoolProperty(name="Fractions", default=False,
                                description="Lets liquid slide smoothly along curved obstacles instead of catching on the grid cells. The domain walls get one cell thicker.")
    fractions_threshold: FloatProperty(name="Obstacle Threshold", default=0.05, min=0.001, max=1.0,
                                       precision=3, description="A grid cell side that is blocked by less than this much counts as open.")
    fractions_distance: FloatProperty(name="Obstacle Distance", default=0.5, min=-5.0, max=5.0,
                                      description="How far, in grid cells, liquid is kept away from obstacles when Fractions is on.")
    use_viscosity: BoolProperty(name="Viscosity", default=False,
                                description="Makes the liquid thick and slow, like honey or lava. Takes longer to bake.")
    viscosity_value: FloatProperty(name="Strength", default=0.05, min=0.0, max=10.0, precision=4,
                                   description="How thick the liquid is. Higher is thicker, like honey instead of water.")
    use_diffusion: BoolProperty(name="Diffusion", default=False,
                                description="Sets the thickness with real-world numbers (Base and Exponent) instead of the simple Strength slider.")
    viscosity_base: FloatProperty(name="Base", default=1.0, min=0.0, max=10.0,
                                  description="The first number of the thickness: Base x 10 to the power of minus Exponent.")
    viscosity_exponent: IntProperty(name="Exponent", default=6, min=0, max=10,
                                    description="The power of ten of the thickness. A smaller number is thicker (water is about 1 x 10^-6).")
    surface_tension: FloatProperty(name="Surface Tension", default=0.0, min=0.0, max=10.0, soft_max=1.0,
                                   precision=3, unit="NONE",
                                   description="Pulls the liquid surface together, so small drops and thin streams round off. "
                                               "In N/m: water is 0.072. Only visible at small scales (a few cm) or with higher "
                                               "values. High Quality only. Bakes take much longer (smaller time steps).")
    output_particles: BoolProperty(name="Write Particles", default=True,
                                   description="Also save the liquid particles for every frame. Not needed for the liquid surface.")
    use_spray_particles: BoolProperty(name="Spray", default=False,
                                      description="Add droplets that fly off the liquid when it splashes.")
    use_foam_particles: BoolProperty(name="Foam", default=False,
                                     description="Add foam that floats on top of the liquid.")
    use_bubble_particles: BoolProperty(name="Bubbles", default=False,
                                       description="Add bubbles that form under the water and rise up.")
    particle_scale: IntProperty(name="Upres", default=1, min=1, max=4,
                                description="How much finer the spray, foam and bubble particles are placed than the simulation. Higher uses more memory.")
    sndparticle_boundary: EnumProperty(name="Boundary", default="DELETE",
                                       items=[("DELETE", "Delete", "Remove them."),
                                              ("PUSHOUT", "Push Out", "Push them back out of the obstacle.")], description="What happens to spray, foam and bubbles that touch an obstacle.")
    sndparticle_antitunneling: EnumProperty(
        name="Thin Walls", default="AUTO",
        items=[("OFF", "Off", "Fast particles can jump through thin obstacles, as in Blender."),
               ("AUTO", "Auto", "Check the path of each particle in steps of half a cell, so none passes through a thin obstacle. Only fast particles cost extra time."),
               ("FIXED", "Fixed Steps", "Check a fixed number of points along each particle's path, as Mantaflow does.")],
        description="Stop spray, foam and bubbles passing through thin obstacles. Only used in High Quality; Classic always behaves like Blender.")
    sndparticle_antitunnel_steps: IntProperty(
        name="Steps", default=4, min=2, max=16,
        description="How many points along each particle's path are checked for an obstacle when Thin Walls is Fixed Steps.")
    sndparticle_sampling: EnumProperty(
        name="Sampling", default="MULTIPLE",
        items=[("SINGLE", "Single", "One spot per cell, as in Blender. Fewer particles, in blocks the size of a cell."),
               ("MULTIPLE", "Multiple", "Eight spots per cell, so particles follow the splash more smoothly. The sampling rates are scaled down automatically, so you get about the same number of particles as Single, but smoother.")],
        description="How many places in each cell can throw off particles. Only used in High Quality; Classic always behaves like Blender.")
    sndparticle_potential_radius: IntProperty(name="Potential Radius", default=2, min=1, max=4,
                                              description="How far around each spot, in cells, the liquid is checked when deciding whether to make particles.")
    sndparticle_potential_min_wavecrest: FloatProperty(name="Wave Crest Min", default=2.0, min=0.0, max=1000.0,
                                                       description="A wave must be at least this sharp to throw off any spray or foam.")
    sndparticle_potential_max_wavecrest: FloatProperty(name="Wave Crest Max", default=8.0, min=0.0, max=1000.0,
                                                       description="A wave this sharp or sharper throws off the most particles.")
    sndparticle_potential_min_trappedair: FloatProperty(name="Trapped Air Min", default=5.0, min=0.0, max=1000.0,
                                                        description="At least this much air must get trapped under the surface before bubbles appear.")
    sndparticle_potential_max_trappedair: FloatProperty(name="Trapped Air Max", default=20.0, min=0.0, max=1000.0,
                                                        description="This much trapped air or more makes the most bubbles.")
    sndparticle_potential_min_energy: FloatProperty(name="Energy Min", default=1.0, min=0.0, max=1000.0,
                                                    description="The liquid must move at least this fast before it makes particles.")
    sndparticle_potential_max_energy: FloatProperty(name="Energy Max", default=5.0, min=0.0, max=1000.0,
                                                    description="Liquid moving this fast or faster makes the most particles.")
    sndparticle_sampling_wavecrest: IntProperty(name="Wave Crest Sampling", default=200, min=0, max=10000,
                                                description="How many particles each wave crest makes. About half of Blender's amount comes out when Volume Correction is on; raise this to match.")
    sndparticle_sampling_trappedair: IntProperty(name="Trapped Air Sampling", default=40, min=0, max=10000,
                                                 description="How many bubbles trapped air makes.")
    sndparticle_bubble_buoyancy: FloatProperty(name="Bubble Buoyancy", default=0.5, min=0.0, max=100.0,
                                               description="How quickly bubbles rise up through the liquid.")
    sndparticle_bubble_drag: FloatProperty(name="Bubble Drag", default=0.6, min=0.0, max=100.0,
                                           description="How strongly the moving liquid carries the bubbles along.")
    sndparticle_life_min: FloatProperty(name="Life Min", default=10.0, min=0.0, max=10000.0,
                                        description="The shortest time a particle lives before it disappears.")
    sndparticle_life_max: FloatProperty(name="Life Max", default=25.0, min=0.0, max=10000.0,
                                        description="The longest time a particle lives before it disappears.")
    sndparticle_width: FloatProperty(name="Point Width", default=0.01, min=0.0001, max=1.0, subtype="DISTANCE",
                                     description="Size of each particle point in the exported USD file, in metres.")
    mesh_generator: EnumProperty(name="Mesh Generator", default="ANISOTROPIC",
                                 items=[("ANISOTROPIC", "High Quality", "Flatter, smoother surfaces, thin splash sheets and fewer blobby drops. Not in Blender."),
                                        ("IMPROVED", "Final", "Smooth surface. Slower, nicer (Blender's Final)."),
                                        ("UNION", "Fast", "Lumpy surface made of overlapping balls. Faster (Blender's Fast).")], description="How the liquid surface is built from the particles.")
    mesh_scale: IntProperty(name="Mesh Upres", default=2, min=1, max=4,
                            description="How much finer the liquid surface is than the simulation. Higher gives smoother detail but uses more memory.")
    mesh_particle_radius: FloatProperty(name="Particle Radius", default=2.0, min=0.0, max=10.0,
                                        description="Size of the particles when building the surface, in cells. Bigger smooths out small bumps.")
    mesh_smoothen_pos: IntProperty(name="Smoothing Positive", default=1, min=0, max=100,
                                   description="How many times to smooth bumps that stick out of the surface.")
    mesh_smoothen_neg: IntProperty(name="Smoothing Negative", default=1, min=0, max=100,
                                   description="How many times to smooth dents that go into the surface.")
    mesh_concave_upper: FloatProperty(name="Concavity Upper", default=3.5, min=0.0, max=10.0,
                                      description="How strongly sharp inward corners in the surface are rounded off (upper limit).")
    mesh_concave_lower: FloatProperty(name="Concavity Lower", default=0.4, min=0.0, max=10.0,
                                      description="How strongly sharp inward corners in the surface are rounded off (lower limit).")
    liquid_threshold: FloatProperty(name="Mesh Threshold", default=0.0, soft_min=-0.1, soft_max=0.1,
                                    description="Moves the liquid surface slightly in or out. 0 is the exact surface.")
    liquid_adaptivity: FloatProperty(name="Mesh Adaptivity", default=0.0, min=0.0, max=1.0,
                                     description="Uses fewer polygons on flat parts of the liquid surface. Higher is lighter but less detailed.")
    resolution_max: IntProperty(name="Resolution Divisions", default=32, min=6, soft_max=512,
                                description="How finely the domain is divided along its longest side. Higher gives more detail but needs much more memory and time.")
    use_2d: BoolProperty(name="2D", default=False,
                         description="Simulate a flat slice instead of a 3D volume: faster, good for testing. Gas only, and the domain's Y size is ignored.")
    time_scale: FloatProperty(name="Speed", default=1.0, min=0.0, soft_max=10.0,
                              description="Speed of the simulation. Below 1 gives slow motion, above 1 speeds it up.")
    cfl_condition: FloatProperty(name="CFL Number", default=4.0, min=0.0, soft_max=10.0,
                                 description="How far the fluid may move in one step, in cells. Lower is more stable but slower; higher is faster but can go wrong.")
    timesteps_min: IntProperty(name="Min Substeps", default=1, min=1, soft_max=10,
                               description="The fewest calculation steps used per frame.")
    timesteps_max: IntProperty(name="Max Substeps", default=4, min=1, soft_max=10,
                               description="The most calculation steps used per frame. Raise it for fast action; it makes the bake slower.")
    gravity: FloatVectorProperty(name="Gravity", default=(0.0, 0.0, -9.81), subtype="ACCELERATION",
                                 description="Gravity used for this fluid when the scene's own gravity is turned off.")
    effector_weights: PointerProperty(type=OpenBillowWeights)
    alpha: FloatProperty(name="Density Buoyancy", default=1.0, soft_min=-5.0, soft_max=5.0,
                         description="How strongly dense smoke sinks. Higher makes it fall faster.")
    beta: FloatProperty(name="Heat Buoyancy", default=1.0, soft_min=-5.0, soft_max=5.0,
                        description="How strongly hot smoke rises. Higher makes it climb faster.")
    vorticity: FloatProperty(name="Vorticity", default=0.0, min=0.0, soft_max=2.0,
                             description="Adds swirling to the smoke. Higher gives more curls.")
    use_dissolve_smoke: BoolProperty(name="Dissolve Smoke", default=False,
                                     description="Let the smoke fade away over time.")
    burning_rate: FloatProperty(name="Reaction Speed", default=0.75, min=0.01, max=4.0,
                                description="How fast the fuel burns. Higher gives shorter, quicker flames.")
    flame_smoke: FloatProperty(name="Smoke", default=1.0, min=0.0, max=8.0,
                               description="How much smoke the flames make.")
    flame_vorticity: FloatProperty(name="Vorticity", default=0.5, min=0.0, max=2.0,
                                   description="Adds swirling to the flames. Higher gives more flickering.")
    flame_ignition: FloatProperty(name="Minimum", default=1.5, min=0.5, max=5.0,
                                  description="The temperature at which the fuel starts to burn.")
    flame_max_temp: FloatProperty(name="Maximum", default=3.0, min=0.5, max=10.0,
                                  description="The hottest the flame can get.")
    quality: EnumProperty(name="Quality", default="HIGH", update=_quality_updated, items=[
        ("CLASSIC", "Classic", "Matches Blender. Smoke: fastest, swirls and fine detail fade out quickly. "
                               "Liquid: the surface snaps to whole cells in the pressure solve."),
        ("HIGH", "High Quality", "Smoke: keeps much more swirl and fine detail at the same resolution; each "
                                 "step takes about 1.7x as long and needs a little more memory. Liquid: "
                                 "the pressure solve sees the exact surface position (smoother waves)")])
    fire_model: EnumProperty(name="Fire Model", default="MANTAFLOW", items=[
        ("MANTAFLOW", "Blender", "Matches Blender's own fire. Soft, blobby flames."),
        ("COMBUSTION", "Combustion", "Fuel burns only when hot enough, gives off heat, cools down and expands. Gives tongues and sharper flame tips.")])
    heat_release: FloatProperty(name="Heat Release", default=12.0, min=0.0, soft_max=40.0,
                                description="How much heat each bit of burning fuel gives off. Higher gives hotter, brighter flames.")
    cooling: FloatProperty(name="Cooling", default=12.0, min=0.0, soft_max=60.0,
                           description="How fast the hot gas cools down. Higher gives shorter flames with more distinct tips.")
    cooling_radiative: FloatProperty(name="Radiative Cooling", default=20.0, min=0.0, soft_max=100.0,
                                     description="Extra cooling that grows quickly with temperature, so the hottest gas loses heat fastest. Keeps flame cores from running away.")
    expansion: FloatProperty(name="Expansion", default=0.1, min=0.0, soft_max=1.0,
                             description="How much the gas swells as it burns. A little makes flames lively; above 1 they balloon sideways.")
    flame_kelvin: FloatProperty(name="Flame Temperature", default=1800.0, min=300.0, max=4000.0, subtype="TEMPERATURE", unit="TEMPERATURE",
                                description="Temperature in Kelvin written to the temperature grid at the hottest flame. Drives the Blackbody colour.")
    ambient_kelvin: FloatProperty(name="Air Temperature", default=293.0, min=0.0, max=1000.0, subtype="TEMPERATURE", unit="TEMPERATURE",
                                  description="Temperature in Kelvin of the unheated air.")
    disturbance: FloatProperty(name="Disturbance", default=60.0, min=0.0, soft_max=300.0,
                               description="Random pushes in the hot gas that break the flame into separate tongues. 0 turns it off.")
    disturbance_size: FloatProperty(name="Size", default=0.1, min=0.001, soft_max=1.0, subtype="DISTANCE", unit="LENGTH",
                                    description="Size of the random pushes. Small is fuzzy, large gives fewer, bigger tongues.")
    disturbance_rate: FloatProperty(name="Rate", default=4.0, min=0.0, soft_max=30.0,
                                    description="How many times per second the random pattern changes.")
    shredding: FloatProperty(name="Shredding", default=0.0, min=0.0, soft_max=40.0,
                             description="Tears the flame edge apart. Subtle in a rising flame; 0 is off.")
    use_noise: BoolProperty(name="High Resolution", default=False,
                            description="Adds fine swirling detail on a finer grid. Looks better, but is slower and needs more memory.")
    noise_upres: IntProperty(name="Upres", default=2, min=1, max=4,
                             description="How much finer the detail grid is than the simulation, in each direction.")
    noise_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=10.0,
                                  description="How strong the added fine detail is.")
    noise_pos_scale: FloatProperty(name="Scale", default=2.0, min=0.0001, max=100.0,
                                   description="Size of the fine detail. Higher gives smaller, finer detail.")
    noise_time_anim: FloatProperty(name="Time", default=0.1, min=0.0001, max=10.0,
                                   description="How quickly the fine detail changes over time.")
    dissolve_speed: IntProperty(name="Time", default=5, min=1, soft_max=10000,
                                description="How many frames the smoke takes to dissolve. Lower is faster.")
    use_dissolve_smoke_log: BoolProperty(name="Slow", default=True,
                                         description="Smoke fades quickly at first and lingers at the end, instead of fading evenly.")
    use_collision_border_front: BoolProperty(name="Front", default=False,
                                             description="Make the front side of the domain a solid wall instead of letting the fluid leave.")
    use_collision_border_back: BoolProperty(name="Back", default=False,
                                            description="Make the back side of the domain a solid wall instead of letting the fluid leave.")
    use_collision_border_right: BoolProperty(name="Right", default=False,
                                             description="Make the right side of the domain a solid wall instead of letting the fluid leave.")
    use_collision_border_left: BoolProperty(name="Left", default=False,
                                            description="Make the left side of the domain a solid wall instead of letting the fluid leave.")
    use_collision_border_top: BoolProperty(name="Top", default=False,
                                           description="Make the top of the domain a solid ceiling instead of letting the fluid leave.")
    use_collision_border_bottom: BoolProperty(name="Bottom", default=False,
                                              description="Make the bottom of the domain a solid floor instead of letting the fluid leave.")
    samples_per_frame: IntProperty(name="Motion Samples", default=4, min=1, max=16,
                                   description="How many times per frame the movement of animated objects is checked. More copes better with fast-moving objects.")
    use_guide: BoolProperty(name="Guides", default=False,
                            description="Steer the fluid so it follows a guide movement, from Guide objects or from another domain's bake.")
    guide_alpha: FloatProperty(name="Weight", default=2.0, min=1.0, max=100.0,
                               description="How strongly the fluid is pulled along. Higher makes the fluid lag further behind the guide.")
    guide_beta: IntProperty(name="Size", default=5, min=1, max=50,
                            description="Size of the swirls the guide makes, in cells. Higher gives larger swirls.")
    guide_vel_factor: FloatProperty(name="Velocity Factor", default=2.0, min=0.0, max=100.0,
                                    description="Makes the guide movement stronger or weaker.")
    guide_source: EnumProperty(name="Velocity Source", default="EFFECTOR", items=[
        ("EFFECTOR", "Effector", "Guide objects in the scene."),
        ("DOMAIN", "Domain", "The finished bake of another OpenBillow domain (the Guide Parent).")], description="Where the guide movement comes from.")
    guide_parent: PointerProperty(name="Guide Parent", type=bpy.types.Object,
                                  poll=lambda self, ob: ob.openbillow.role == "DOMAIN" and ob != self.id_data,
                                  description="The domain whose finished bake guides this one. Bake that one first.")
    use_scene_range: BoolProperty(name="Scene Frame Range", default=True,
                                  description="Bake the scene's start-to-end frames. Turn off to pick your own range.")
    frame_start: IntProperty(name="Start", default=1, description="First frame to bake.")
    frame_end: IntProperty(name="End", default=250, description="Last frame to bake.")
    material: PointerProperty(name="Material", type=bpy.types.Material,
                              description="Material for the smoke or liquid. Presets set it. Empty uses the OpenBillow default.")
    cache_dir: StringProperty(name="Cache", subtype="DIR_PATH", default="", options={"PATH_SUPPORTS_BLEND_RELATIVE"},
                              description="Folder where the baked frames are saved. Empty uses a folder per domain next to the .blend file.")
    refresh_vdb: BoolProperty(name="Refresh VDB on Bake", default=True,
                              description="When a bake finishes, make Blender reload the baked files. Needed after baking again over old frames.")


class OpenBillowFlow(PropertyGroup):
    behavior: EnumProperty(name="Flow Behavior", default="GEOMETRY", items=[
        ("GEOMETRY", "Liquid", "Starts out as liquid: fills this shape on the first frame."),
        ("INFLOW", "Inflow", "Keeps pouring new liquid in."), ("OUTFLOW", "Outflow", "Drains away any liquid that touches it.")], description="What this object does to the liquid.")
    liquid_surface_distance: FloatProperty(name="Surface Emission", default=0.0, min=0.0, max=10.0,
                                           description="How far from the surface, in cells, liquid is made. 0 is right at the surface.")
    use_initial_velocity: BoolProperty(name="Initial Velocity", default=False,
                                       description="Give the new liquid a starting speed and direction.")
    velocity_factor: FloatProperty(name="Source", default=1.0, min=0.0, max=100.0,
                                   description="How much of the object's own movement the new liquid picks up. 1 is all of it.")
    velocity_normal: FloatProperty(name="Normal", default=0.0, min=-100.0, max=100.0,
                                   description="Speed at which the liquid is pushed straight out of the object's surface.")
    velocity_coord: FloatVectorProperty(name="Initial", default=(0.0, 0.0, 0.0), subtype="VELOCITY",
                                        description="Starting speed and direction of the new liquid, in metres per second.")
    flow_type: EnumProperty(name="Flow Type", default="SMOKE", items=[
        ("SMOKE", "Smoke", "Smoke only."), ("FIRE", "Fire", "Fire only."), ("BOTH", "Fire + Smoke", "Fire and smoke.")], description="What this object produces.")
    fuel_amount: FloatProperty(name="Flame Rate", default=1.0, min=0.0, max=5.0,
                               description="How much fuel the source makes. More fuel gives bigger, longer-burning flames.")
    use_emit_range: BoolProperty(name="Limit Emission", default=False,
                                 description="Only make smoke or fire between two frames. A short burst makes an explosion.")
    emit_start: IntProperty(name="Start", default=1, description="First frame it emits.")
    emit_end: IntProperty(name="End", default=1, min=1, description="Last frame it emits.")
    density: FloatProperty(name="Density", default=1.0, min=0.0, max=1.0,
                           description="How thick the smoke it makes is.")
    temperature: FloatProperty(name="Temperature", default=1.0, min=-5.0, max=5.0,
                               description="How hot the smoke or fire it makes is. Hotter rises faster.")
    surface_distance: FloatProperty(name="Surface Emission", default=1.5, min=0.0, max=10.0,
                                    description="How far from the surface, in cells, smoke is made. Higher gives a thicker source.")
    volume_density: FloatProperty(name="Volume Emission", default=0.0, min=0.0, max=1.0,
                                  description="How much smoke is made inside the object, not only on its surface.")
    use_plane_init: BoolProperty(name="Is Planar", default=False,
                                 description="Tick for a flat object such as a plane, which has no inside, so it is treated as a surface only.")


class OpenBillowObstacle(PropertyGroup):
    surface_distance: FloatProperty(name="Thickness", default=0.0, min=0.0, max=10.0,
                                    description="Makes the obstacle thicker by this many cells. 0 is its exact shape.")
    use_plane_init: BoolProperty(name="Is Planar", default=False,
                                 description="Tick for a flat object such as a plane, which has no inside, so it is treated as a surface only.")


class OpenBillowGuide(PropertyGroup):
    guide_mode: EnumProperty(name="Guide Mode", default="OVERRIDE", items=[
        ("OVERRIDE", "Override", "Use this object's movement as it is."),
        ("AVERAGED", "Averaged", "Use half of this object's movement."),
        ("MAXIMUM", "Maximize", "Use the strongest part of the movement."),
        ("MINIMUM", "Minimize", "Use none of it: no guidance.")], description="How this object's movement is used to steer the fluid.")
    velocity_factor: FloatProperty(name="Velocity Factor", default=1.0, min=-100.0, max=100.0,
                                   description="Makes this object's movement a stronger or weaker guide.")
    surface_distance: FloatProperty(name="Surface Thickness", default=0.0, min=0.0, max=10.0,
                                    description="Makes the guide thicker by this many cells. 0 is its exact shape.")
    use_plane_init: BoolProperty(name="Is Planar", default=False,
                                 description="Tick for a flat object such as a plane, which has no inside, so it is treated as a surface only.")


class OpenBillowObject(PropertyGroup):
    role: EnumProperty(name="Type", default="NONE", items=[
        ("NONE", "None", "Not part of the simulation."), ("DOMAIN", "Domain", "The box the fluid is simulated in."),
        ("FLOW", "Flow", "Makes smoke, fire or liquid."), ("OBSTACLE", "Obstacle", "Blocks the fluid."),
        ("GUIDE", "Guide", "Its movement steers the fluid. It does not block it.")], description="What this object is in the fluid simulation.")
    domain: PointerProperty(type=OpenBillowDomain)
    flow: PointerProperty(type=OpenBillowFlow)
    obstacle: PointerProperty(type=OpenBillowObstacle)
    guide: PointerProperty(type=OpenBillowGuide)


class OpenBillowPreferences(AddonPreferences):
    bl_idname = __package__

    cache_root: StringProperty(name="Cache Root", subtype="DIR_PATH", default="", options={"PATH_SUPPORTS_BLEND_RELATIVE"},
                               description="Folder where baked frames are saved, one subfolder per domain. Empty puts them next to the .blend file.")
    device_mode: EnumProperty(name="Device", default="AUTO", description="Where bakes run. Auto picks the fastest option that fits in memory.", items=[
        ("AUTO", "Auto", "Bake on the graphics card; when it does not fit, page through system RAM (about 2x slower), else use the CPU"),
        ("GPU", "GPU", "Always bake on the graphics card; refuse bakes that do not fit"),
        ("CPU", "CPU", "Always bake on the CPU. Much slower, but uses system RAM, so larger bakes fit")])
    device: StringProperty(name="GPU", default="",
                           description="Which graphics card to use, for example cuda:0. Empty uses the default one.")
    vram_limit_mb: IntProperty(name="VRAM Limit (MB)", default=0, min=0,
                               description="Refuse to start a bake that is estimated to need more graphics memory than this. 0 uses what the card has, minus 1 GB.")
    device_total_mb: IntProperty(default=0, options={"HIDDEN"})
    device_name: StringProperty(default="", options={"HIDDEN"})

    support_key: StringProperty(name="Licence Key", subtype="PASSWORD", default="",
                                description="Your licence key from your purchase (Gumroad), or the OB- key from the claim page.")
    accept_terms: BoolProperty(name="I accept the Licence and Terms of Service", default=False,
                               description="Required to activate your licence key. Read the Licence and the Terms of Service on the OpenBillow website first (buttons above).")
    activated_for: StringProperty(default="", options={"HIDDEN"})  # activation.token() of the activated key + email + computer
    support_email: StringProperty(name="Email", default="",
                                  description="Where support replies. Your key is tied to the first email address you use with it.")
    support_description: StringProperty(name="Problem", default="", options={"SKIP_SAVE"},
                                        description="What went wrong, in a sentence or two.")
    support_attach_log: BoolProperty(name="Attach last bake log", default=True,
                                     description="Include the end of the bake log in your request. Your home folder name is hidden.")

    support_attach_errors: BoolProperty(name="Attach recent errors", default=True,
                                        description="Include the last 20 warnings and errors OpenBillow saw. Your home folder name is hidden.")

    def draw(self, context):
        from .flavor import STORE
        col = self.layout.column()
        col.prop(self, "cache_root")
        col.prop(self, "device_mode")
        sub = col.column()
        sub.enabled = self.device_mode != "CPU"
        sub.prop(self, "device")
        sub.prop(self, "vram_limit_mb")
        row = col.row()
        from . import ops
        if ops.warp_available():
            row.operator("openbillow.detect_gpu", icon="FILE_REFRESH")
            row.label(text=f"{self.device_name} ({self.device_total_mb} MB)" if self.device_total_mb
                      else "GPU not detected yet")
        else:
            if not STORE:
                row.operator("openbillow.install_warp", icon="IMPORT")
            row.label(text="GPU engine (Warp) not installed", icon="ERROR")
        from . import diagnostics
        box = self.layout.box()
        box.label(text="Bug reports", icon="ERROR")
        diagnostics.draw_report(box)
        box.prop(self, "support_attach_errors", text="Include recent OpenBillow warnings")
        if not STORE:
            from . import presets_online, support, updates
            support.draw_box(self.layout)
            presets_online.draw_box(self.layout)
            updates.draw_box(self.layout)

    def device_arg(self):
        """--device value for the CLI."""
        if self.device_mode == "CPU":
            return "cpu"
        if self.device_mode == "GPU":
            return self.device.strip() or "cuda"
        return "auto:" + self.device.strip() if self.device.strip() else "auto"

    def limit_mb(self):
        from .openbillow import estimate
        if self.vram_limit_mb:
            return float(self.vram_limit_mb)
        if self.device_total_mb:
            return estimate.default_limit_mb(self.device_total_mb * 2**20)
        return None


def prefs():
    return bpy.context.preferences.addons[__package__].preferences


_classes = (OpenBillowWeights, OpenBillowDomain, OpenBillowFlow, OpenBillowObstacle, OpenBillowGuide, OpenBillowObject,
            OpenBillowPreferences)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Object.openbillow = PointerProperty(type=OpenBillowObject)


def unregister():
    del bpy.types.Object.openbillow
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
