use crate::graphics::state::DISP_COEF;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl ColorVertex {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ColorVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub const COLOR_VERTICES: &[ColorVertex] = &[
    ColorVertex { position: [0.0, 0.0, 0.0], color: [1.0, 0.0, 0.0, 1.0]},
    ColorVertex { position: [1.0, 0.0, 0.0], color: [1.0, 1.0, 0.0, 1.0], },
    ColorVertex { position: [1.0, 1.0, 0.0], color: [1.0, 0.0, 1.0, 1.0], },
    ColorVertex { position: [0.0, 1.0, 0.0], color: [0.0, 0.0, 1.0, 1.0], },
];
