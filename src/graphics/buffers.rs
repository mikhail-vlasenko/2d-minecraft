use cgmath::Rotation3;
use wgpu::Device;
use wgpu::util::DeviceExt;
use crate::graphics::instance::Instance;
use crate::graphics::state::{DISP_COEF, INITIAL_POS, TILES_PER_ROW};
use crate::graphics::vertex::{INDICES, PLAYER_VERTICES, VERTICES};


/// Creates and stores wgpu buffers
pub struct Buffers {
    pub vertex_buffer: wgpu::Buffer,
    pub player_vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub player_instance_buffer: wgpu::Buffer,
}

impl Buffers {
    pub fn new(device: &Device) -> Self {
        let instances = (0..TILES_PER_ROW).flat_map(|y| {
            (0..TILES_PER_ROW).map(move |x| {
                let position =
                    cgmath::Vector3 { x: x as f32 * DISP_COEF, y: y as f32 * DISP_COEF, z: 0.0 }
                        + INITIAL_POS;

                let rotation =
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0));

                Instance {
                    position,
                    rotation,
                    scaling: DISP_COEF,
                }
            })
        }).collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let player_instances = (0..4).map(move |angle| {
            let x_rot_compensation = if angle == 1 || angle == 2 { 1 } else { 0 };
            let y_rot_compensation = if angle > 1 { 1 } else { 0 };
            let position =
                cgmath::Vector3 {
                    x: (((TILES_PER_ROW - 1) / 2) + x_rot_compensation) as f32 * DISP_COEF,
                    y: (((TILES_PER_ROW - 1) / 2) + y_rot_compensation) as f32 * DISP_COEF,
                    z: 0.0
                }
                    + INITIAL_POS;

            let rotation =
                cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(90.0 * angle as f32));

            Instance {
                position,
                rotation,
                scaling: DISP_COEF,
            }
        }).collect::<Vec<_>>();

        let player_instance_data = player_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let player_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&player_instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let player_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Player Buffer"),
                contents: bytemuck::cast_slice(PLAYER_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        Self {
            vertex_buffer,
            player_vertex_buffer,
            index_buffer,
            instance_buffer,
            player_instance_buffer,
        }
    }
}