//! Minimal OpenGL renderer for the 3D preview, drawn through an egui paint callback.

use std::sync::{Arc, Mutex};

use eframe::egui_glow;
use eframe::glow::{self, HasContext};
use glam::{Mat4, Vec3};

use crate::mesh::Mesh;

pub const KIND_LITHOPHANE: f32 = 0.0;
pub const KIND_SUPPORT: f32 = 1.0;
pub const KIND_BED: f32 = 2.0;

/// Floats per vertex: position (3), normal (3), kind, thickness.
const STRIDE: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            yaw: -1.2,
            pitch: 0.35,
            distance: 300.0,
            target: Vec3::new(0.0, 0.0, 70.0),
        }
    }
}

impl Camera {
    const FOV_Y: f32 = 0.7;

    pub fn fit(&mut self, lo: Vec3, hi: Vec3) {
        self.target = (lo + hi) * 0.5;
        let radius = ((hi - lo).length() * 0.5).max(1.0);
        self.distance = radius / (Self::FOV_Y * 0.5).sin() * 1.05;
    }

    pub fn eye(&self) -> Vec3 {
        let dir = Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        );
        self.target + dir * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let near = (self.distance * 0.01).max(0.1);
        let far = self.distance * 10.0 + 1000.0;
        let proj =
            glam::camera::rh::proj::opengl::perspective(Self::FOV_Y, aspect.max(0.01), near, far);
        proj * glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Z)
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.5, 1.5);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, view_height_px: f32) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward);
        let mm_per_px = 2.0 * self.distance * (Self::FOV_Y * 0.5).tan() / view_height_px.max(1.0);
        self.target += (-right * dx + up * dy) * mm_per_px;
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.002).exp()).clamp(5.0, 5000.0);
    }
}

/// Vertex buffer contents and the draw calls that place the meshes.
pub struct SceneData {
    vertices: Vec<f32>,
    /// First vertex, vertex count and offset of each draw call.
    draws: Vec<(i32, i32, Vec3)>,
    pub bounds: Option<(Vec3, Vec3)>,
}

/// Flattens meshes into a non-indexed vertex buffer with face normals. Each mesh is stored
/// once and drawn at each of its offsets, with a build plate under everything.
pub fn scene_data(meshes: &[(&Mesh, f32, &[Vec3])]) -> SceneData {
    let count: usize = meshes.iter().map(|(m, _, _)| m.triangles.len() * 3).sum();
    let mut vertices = Vec::with_capacity(count * STRIDE + 6 * STRIDE);
    let mut draws = Vec::new();
    let mut bounds: Option<(Vec3, Vec3)> = None;
    for (mesh, kind, offsets) in meshes {
        let first = vertices.len() / STRIDE;
        for t in &mesh.triangles {
            let n = mesh.triangle_normal(t).as_vec3();
            for &v in t {
                let p = mesh.positions[v as usize];
                let thickness = mesh.attr.get(v as usize).copied().unwrap_or(-1.0);
                vertices.extend_from_slice(&[p[0], p[1], p[2], n.x, n.y, n.z, *kind, thickness]);
            }
        }
        let n = vertices.len() / STRIDE - first;
        let Some((lo, hi)) = mesh.bbox().filter(|_| n > 0) else {
            continue;
        };
        for &o in offsets.iter() {
            let (lo, hi) = (lo.as_vec3() + o, hi.as_vec3() + o);
            bounds = Some(bounds.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            draws.push((first as i32, n as i32, o));
        }
    }
    if let Some((lo, hi)) = bounds {
        // Build plate under the model.
        let m = (hi - lo).max_element() * 0.15 + 10.0;
        let (a, b) = (lo - Vec3::splat(m), hi + Vec3::splat(m));
        let z = -0.05;
        draws.push(((vertices.len() / STRIDE) as i32, 6, Vec3::ZERO));
        for (x, y) in [
            (a.x, a.y),
            (b.x, a.y),
            (b.x, b.y),
            (a.x, a.y),
            (b.x, b.y),
            (a.x, b.y),
        ] {
            vertices.extend_from_slice(&[x, y, z, 0.0, 0.0, 1.0, KIND_BED, -1.0]);
        }
    }
    SceneData {
        vertices,
        draws,
        bounds,
    }
}

#[derive(Clone, Copy)]
pub struct Uniforms {
    pub view_proj: Mat4,
    pub eye: Vec3,
    pub backlit: bool,
    pub t_range: [f32; 2],
}

#[derive(Default)]
pub struct Scene {
    program: Option<glow::Program>,
    vao: Option<glow::VertexArray>,
    vbo: Option<glow::Buffer>,
    draws: Vec<(i32, i32, Vec3)>,
    pending: Option<SceneData>,
    pub error: Option<String>,
}

pub type SharedScene = Arc<Mutex<Scene>>;

impl Scene {
    pub fn set_data(&mut self, data: SceneData) {
        self.pending = Some(data);
    }

    fn init(&mut self, gl: &glow::Context) -> Result<(), String> {
        let version = egui_glow::ShaderVersion::get(gl);
        if !version.is_new_shader_interface() {
            return Err(format!(
                "OpenGL shader version {version:?} is too old for the 3D preview"
            ));
        }
        let header = format!(
            "{}{}",
            version.version_declaration(),
            if version.is_embedded() {
                "precision highp float;\n"
            } else {
                ""
            }
        );
        let vs = r#"
            in vec3 a_pos;
            in vec3 a_normal;
            in vec2 a_attr;
            uniform mat4 u_view_proj;
            uniform vec3 u_offset;
            out vec3 v_normal;
            out vec3 v_pos;
            out vec2 v_attr;
            void main() {
                v_normal = a_normal;
                v_pos = a_pos + u_offset;
                v_attr = a_attr;
                gl_Position = u_view_proj * vec4(v_pos, 1.0);
            }
        "#;
        let fs = r#"
            in vec3 v_normal;
            in vec3 v_pos;
            in vec2 v_attr;
            uniform vec3 u_eye;
            uniform float u_backlit;
            uniform vec2 u_t_range;
            out vec4 out_color;
            void main() {
                vec3 n = normalize(v_normal);
                vec3 to_eye = normalize(u_eye - v_pos);
                if (dot(n, to_eye) < 0.0) n = -n;
                float kind = v_attr.x;
                vec3 base = kind < 0.5 ? vec3(0.93, 0.92, 0.88)
                          : kind < 1.5 ? vec3(0.35, 0.62, 0.95)
                          : vec3(0.22, 0.23, 0.25);
                vec3 key = normalize(vec3(-0.4, -0.8, 0.9));
                vec3 fill = normalize(vec3(0.7, 0.5, 0.3));
                float light = 0.28 + 0.62 * max(dot(n, key), 0.0) + 0.22 * max(dot(n, fill), 0.0);
                vec3 c = base * light;
                if (u_backlit > 0.5 && kind < 0.5) {
                    float t = v_attr.y;
                    if (t >= 0.0) {
                        float x = clamp((t - u_t_range.x) / max(u_t_range.y - u_t_range.x, 1e-4), 0.0, 1.0);
                        c = vec3(1.0, 0.94, 0.82) * exp(-2.5 * x);
                    } else {
                        c = vec3(1.0, 0.94, 0.82) * 0.25;
                    }
                }
                out_color = vec4(pow(c, vec3(1.0 / 2.2)), 1.0);
            }
        "#;
        unsafe {
            let program = gl.create_program()?;
            let mut shaders = Vec::new();
            for (kind, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
                let shader = gl.create_shader(kind)?;
                gl.shader_source(shader, &format!("{header}{src}"));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    return Err(format!(
                        "shader compile error: {}",
                        gl.get_shader_info_log(shader)
                    ));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }
            gl.bind_attrib_location(program, 0, "a_pos");
            gl.bind_attrib_location(program, 1, "a_normal");
            gl.bind_attrib_location(program, 2, "a_attr");
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(format!(
                    "shader link error: {}",
                    gl.get_program_info_log(program)
                ));
            }
            for s in shaders {
                gl.detach_shader(program, s);
                gl.delete_shader(s);
            }
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            let stride = (STRIDE * 4) as i32;
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, stride, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, stride, 12);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, stride, 24);
            gl.bind_vertex_array(None);
            self.program = Some(program);
            self.vao = Some(vao);
            self.vbo = Some(vbo);
        }
        Ok(())
    }

    pub fn paint(&mut self, gl: &glow::Context, u: &Uniforms) {
        if self.error.is_some() {
            return;
        }
        if self.program.is_none()
            && let Err(e) = self.init(gl)
        {
            self.error = Some(e);
            return;
        }
        let (Some(program), Some(vao), Some(vbo)) = (self.program, self.vao, self.vbo) else {
            return;
        };
        unsafe {
            if let Some(data) = self.pending.take() {
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                let v = &data.vertices;
                let bytes = std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4);
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);
                gl.bind_buffer(glow::ARRAY_BUFFER, None);
                self.draws = data.draws;
            }
            if self.draws.is_empty() {
                return;
            }
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.use_program(Some(program));
            let loc = |name: &str| gl.get_uniform_location(program, name);
            gl.uniform_matrix_4_f32_slice(
                loc("u_view_proj").as_ref(),
                false,
                &u.view_proj.to_cols_array(),
            );
            gl.uniform_3_f32(loc("u_eye").as_ref(), u.eye.x, u.eye.y, u.eye.z);
            gl.uniform_1_f32(loc("u_backlit").as_ref(), if u.backlit { 1.0 } else { 0.0 });
            gl.uniform_2_f32(loc("u_t_range").as_ref(), u.t_range[0], u.t_range[1]);
            gl.bind_vertex_array(Some(vao));
            let offset = loc("u_offset");
            for &(first, count, o) in &self.draws {
                gl.uniform_3_f32(offset.as_ref(), o.x, o.y, o.z);
                gl.draw_arrays(glow::TRIANGLES, first, count);
            }
            gl.bind_vertex_array(None);
            gl.disable(glow::DEPTH_TEST);
        }
    }

    pub fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some(p) = self.program.take() {
                gl.delete_program(p);
            }
            if let Some(v) = self.vao.take() {
                gl.delete_vertex_array(v);
            }
            if let Some(b) = self.vbo.take() {
                gl.delete_buffer(b);
            }
        }
    }
}
