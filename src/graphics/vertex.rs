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

pub const INDICES: &[u16] = &[
    0, 1, 2,
    2, 3, 0,
];

pub const PLAYER_VERTICES: &[Vertex] = &[
    // player square, always in one place
    Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 1.0], },
    Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 1.0], },
    Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 0.0], },
    Vertex { position: [0.0, 1.0, 0.0], tex_coords: [0.0, 0.0], },
];

pub const NIGHT_FILTER_VERTICES: &[Vertex] = &[
    Vertex { position: [-1.0, -1.0, 0.0], tex_coords: [0.0, 1.0], },
    Vertex { position: [1.0, -1.0, 0.0], tex_coords: [1.0, 1.0], },
    Vertex { position: [1.0, 1.0, 0.0], tex_coords: [1.0, 0.0], },
    Vertex { position: [-1.0, 1.0, 0.0], tex_coords: [0.0, 0.0], },
];

pub const HP_VERTICES_SCALING_COEF: f32 = 1.0 / 12.0;
pub const HP_BAR_SCALING_COEF: f32 = 1.0 / 8.0;

pub fn make_hp_vertices(hp_share: f32) -> [Vertex; 4] {
    let hp_share = hp_share.max(0.0).min(1.0);
    [
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 1.0], },
        Vertex { position: [hp_share, 0.0, 0.0], tex_coords: [1.0, 1.0], },
        Vertex { position: [hp_share, HP_VERTICES_SCALING_COEF, 0.0], tex_coords: [1.0, 0.0], },
        Vertex { position: [0.0, HP_VERTICES_SCALING_COEF, 0.0], tex_coords: [0.0, 0.0], },
    ]
}
