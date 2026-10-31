this app can be used to create a lithophane. 


a .stl can be loaded to define the max boundries of the lithophane. 
an image can be loaded that then manipulates the geometry depending on the brighntess of a pixel. 

the image can be edited with simple image edit tools, lightness, blackpoint, whitepoint, contrast, gamma. 

the lithophane can be exported as a .stl file.


tech stack is RUST egui and available rust libraries for generating 3d geometry.

the downloadable   .stl contains :
    - the lithophane part
    - two support parts 



since it is very difficult for a printer to print a lithophane vertical there needs to be supportive geometry. see example_vertical_lithophane and reverse engeneer how it can be done.

the example lithophane is a rectangle but this app will take any stl polygon extruded shape as a constraint for the lithophane result shape.



---

## Building and running

```sh
cargo run --release
```

Needs a Rust toolchain and OpenGL 3.x. File dialogs use GTK3.

The working state (settings, open files, current project) is restored on the next start,
saved or not. Files can also be dragged onto the window: `.lithoproj` files open as a
project, `.stl` files become the outline, and anything else is loaded as the image.

## Using the app

1. **Outline**: pick a rectangle, an ellipse, or *Open outline STL…*. The STL is
   treated as an extruded polygon: the extrusion axis is detected automatically and can be
   overridden. The mid-height cross-section, holes included, becomes the lithophane's
   outline at the STL's real size. Use scale, rotate and mirror to adjust it.
2. **Image**: *Open image…*, then adjust lightness, contrast, gamma, black point
   and white point (with a histogram), plus invert, rotate and mirror. Choose cover/contain/stretch,
   then drag and scroll in the *Framing* view to place the picture inside the outline.
3. **Lithophane**: the number of lithophanes (1–10, see below) and their spacing,
   min thickness (white) and max thickness (black), the width of the solid rim along the
   outline, and the export grid resolution.
4. **Supports**: see below. All dimensions can be changed.
5. **Export STL…** writes one binary STL with a body for each lithophane and for the
   left and right support parts. Each part can also be saved to its own file
   (`…_lithophane.stl`, or `…_lithophane_1.stl` to `…_lithophane_N.stl` for several).

Views: *Framing* (image placement and the outline), *Backlit preview* (simulated
transmitted light), and *3D model* (orbit, pan, zoom; can also simulate backlight).

### Projects

The *Project* menu has *New* (Ctrl+N), *Open…* (Ctrl+O), *Open recent*, *Save* (Ctrl+S)
and *Save as…* (Ctrl+Shift+S). A project (`.lithoproj`, JSON) stores every setting and
refers to the image and outline STL; the files themselves are not copied into it. Each
reference is stored relative to the project file and as an absolute path. Opening tries the
relative path first, so a folder holding the project and its files can be moved or copied as
a whole. A referenced file that can't be found is reported, and the project opens without it.

The window title shows the project name, with `*` while there are unsaved changes. Before
*New* or *Open* would discard them, you are asked whether to save. Running
`lithophane-creator project.lithoproj` opens a project directly.

### Headless export

```sh
lithophane-creator --image photo.jpg --outline heart.stl --out heart_litho.stl --res 0.2 --preview heart.png
lithophane-creator --project heart.lithoproj --out heart_litho.stl   # other options override the project
lithophane-creator --image photo.jpg --out four.stl --count 4 --spacing 3   # four copies in a row
lithophane-creator --help
```

## How the supports work (reverse-engineered from `example_vertical_lithophane`)

The example 3MF (Bambu "Lithophane Maker") holds a 104 × 140 mm relief with a flat back
(0.8–3.2 mm thick, 0.12 mm grid), a 2 mm rim at full thickness, and four "scaffolding"
pieces. Each piece is a 0.8 mm thick triangular buttress wall standing perpendicular to the
plate, flush with its left or right edge, on the front or back face. The walls are
144 mm tall and about 28.5 mm deep at the bed, narrowing to 7.3 mm at the top. They stand
0.01 mm off the plate so they snap away. Each wall has a 2 mm stiffening rib, a 60° fillet
at its base, and a 0.4 mm foot plate on the bed. The foot keeps a 1.4 mm gap to the
lithophane.

This app builds the same parts, with three generalizations for arbitrary outlines:

* the front and back foot plates of each side are joined around the end of the plate,
  so each side is **one support part** (hence two support parts);
* the walls go where the outline is at least 75% of its full height (*Auto*),
  or at a manual inset. Where a wall touches the picture, a narrow strip is made full
  thickness so the wall rests on a flat face. The Framing view shows these strips in blue;
* if the outline has no flat bottom (circle, heart, ...), a **cradle** fills the space
  between the bed and the outline's lower edge. Like slicer support, it keeps a gap
  (default 0.05 mm) below the edge and sideways along slopes. It is split into mirror-image
  left and right halves, one for each support part.

### Several lithophanes

With *Number of lithophanes* above 1, identical copies stand one behind the other, all with
the relief facing the same way, *Spacing* apart (measured between their full-thickness faces,
default 3 mm). They still share **two support parts**, which get deeper to hold the whole row:

* the buttress walls stand only on the front of the first and the back of the last lithophane;
* in each gap between two lithophanes, a full-height spacer wall touches both contact strips
  (with the contact gap), so the row braces itself;
* the foot plate reaches under every spacer like a comb. Its teeth keep the foot gap to the
  lithophanes where the spacing leaves them at least 1 mm, and less otherwise;
* each lithophane gets its own cradle if the outline needs one.

The lithophane is built once and placed at each position, so the preview stays as fast as for
one lithophane. The export grows with the count, about 0.8 M triangles per 108 × 144 mm copy at
0.2 mm.

Overhangs inside the outline that the cradle can't reach (for example inside a "C" shape) still
need slicer supports.

## Code layout

| File | Purpose |
|------|---------|
| `src/shape.rs` | 2D outlines, STL extrusion-axis detection and slicing |
| `src/imaging.rs` | grayscale images and brightness adjustments |
| `src/lithophane.rs` | heightmap sampling; marching-squares clipping to a watertight solid |
| `src/supports.rs` | support walls, ribs, fillets, foot plates and cradle |
| `src/mesh.rs` | meshes, earcut triangulation, extrusion, STL read/write |
| `src/model.rs` | settings and assembly of the printable model |
| `src/project.rs` | project files (save/load, relative file references) |
| `src/viewer.rs` | OpenGL 3D preview |
| `src/app.rs` | egui user interface |

`cargo test` checks among other things that the generated lithophane is watertight for
rectangles, ellipses, outlines with holes and islands, and sub-resolution features.
