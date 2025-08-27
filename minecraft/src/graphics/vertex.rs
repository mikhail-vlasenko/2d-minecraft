#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

pub const VERTICES: &[Vertex] = &[
    Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 1.0], },
    Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 1.0], },
    Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 0.0], },
    Vertex { position: [0.0, 1.0, 0.0], tex_coords: [0.0, 0.0], },
];

pub const CENTERED_SQUARE_VERTICES: &[Vertex] = &[
    Vertex { position: [-0.5, -0.5, 0.0], tex_coords: [0.0, 1.0], },
    Vertex { position: [0.5, -0.5, 0.0], tex_coords: [1.0, 1.0], },
    Vertex { position: [0.5, 0.5, 0.0], tex_coords: [1.0, 0.0], },
    Vertex { position: [-0.5, 0.5, 0.0], tex_coords: [0.0, 0.0], },
];

pub const INDICES: &[u16] = &[
    0, 1, 2,
    2, 3, 0,
];

pub const NIGHT_FILTER_VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 1.0], },
    Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 1.0], },
    Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 0.0], },
    Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 0.0], },
];

const HP_VERTICES_SCALING_COEF: f32 = 1.0 / 12.0;
pub const HP_BAR_SCALING_COEF: f32 = 1.0 / 8.0;

pub fn make_hp_vertices(hp_share: f32) -> [Vertex; 4] {
    let hp_share = hp_share.max(0.0).min(1.0);
    [
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 1.0], },
        Vertex { position: [hp_share, 0.0, 0.0], tex_coords: [hp_share, 1.0], },
        Vertex { position: [hp_share, HP_VERTICES_SCALING_COEF, 0.0], tex_coords: [hp_share, 0.0], },
        Vertex { position: [0.0, HP_VERTICES_SCALING_COEF, 0.0], tex_coords: [0.0, 0.0], },
    ]
}

pub fn make_animation_vertices(frame_number: u32, total_frames: u32) -> [Vertex; 4] {
    let frame_number = frame_number as f32;
    let total_frames = total_frames as f32;
    let frame_share = frame_number / total_frames;
    [
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [frame_share, 1.0], },
        Vertex { position: [1.0, 0.0, 0.0], tex_coords: [frame_share + 1. / total_frames, 1.0], },
        Vertex { position: [1.0, 1.0, 0.0], tex_coords: [frame_share + 1. / total_frames, 0.0], },
        Vertex { position: [0.0, 1.0, 0.0], tex_coords: [frame_share, 0.0], },
    ]
}

// only for the arrow, as there are no other projectiles (yet)
const PROJECTILE_VERTICES_SCALING_COEF: f32 = 1.0 / 8.0;

pub const PROJECTILE_ARROW_VERTICES: &[Vertex] = &[
    Vertex { position: [-PROJECTILE_VERTICES_SCALING_COEF, -0.5, 0.0], tex_coords: [0.0, 1.0], },
    Vertex { position: [PROJECTILE_VERTICES_SCALING_COEF, -0.5, 0.0], tex_coords: [1.0, 1.0], },
    Vertex { position: [PROJECTILE_VERTICES_SCALING_COEF, 0.5, 0.0], tex_coords: [1.0, 0.0], },
    Vertex { position: [-PROJECTILE_VERTICES_SCALING_COEF, 0.5, 0.0], tex_coords: [0.0, 0.0], },
];
