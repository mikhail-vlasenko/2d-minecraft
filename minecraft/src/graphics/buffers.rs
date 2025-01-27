use cgmath::Rotation3;
use wgpu::{Buffer, Device};
use wgpu::util::DeviceExt;
use crate::graphical_config::CONFIG;
use crate::graphics::instance::{Instance, InstanceRaw};
use crate::graphics::vertex::{INDICES, NIGHT_FILTER_VERTICES, PLAYER_VERTICES, VERTICES};


/// Creates and stores wgpu buffers
pub struct Buffers {
    pub vertex_buffer: Buffer,
    pub player_vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub instance_buffer: Buffer,
    pub player_instance_buffer: Buffer,
    pub night_vertex_buffer: Buffer,
    pub night_instance_buffer: Buffer,
    pub map_instance_buffer: Buffer,
    pub hp_bar_vertex_buffers: Vec<Buffer>,
    pub hp_bar_instance_buffer: Buffer,
    pub animation_vertex_buffers: Vec<Buffer>,
    pub projectile_vertex_buffers: Vec<Buffer>,
    pub projectile_instance_buffer: Buffer,
}

impl Buffers {
    pub fn new(device: &Device, map_render_distance: i32) -> Self {
        let instances = (0..4).flat_map(|angle| {
            let x_rot_compensation = if angle == 1 || angle == 2 { 1 } else { 0 };
            let y_rot_compensation = if angle > 1 { 1 } else { 0 };
            (0..CONFIG.tiles_per_row).flat_map(move |y| {
                (0..CONFIG.tiles_per_row).map(move |x| {
                    let position =
                        cgmath::Vector3 {
                            x: (x + x_rot_compensation) as f32 * CONFIG.disp_coef,
                            y: (y + y_rot_compensation) as f32 * CONFIG.disp_coef,
                            z: 0.0
                        }
                            + INITIAL_POS;

                    let rotation =
                        cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(90.0 * angle as f32));

                    Instance {
                        position,
                        rotation,
                        scaling: CONFIG.disp_coef,
                    }
                })
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
            let position =
                cgmath::Vector3 {
                    x: (CONFIG.render_distance as f32 + 0.5) * CONFIG.disp_coef,
                    y: (CONFIG.render_distance as f32 + 0.5) * CONFIG.disp_coef,
                    z: 0.0,
                }
                    + INITIAL_POS;

            let rotation =
                cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(90.0 * angle as f32));

            Instance {
                position,
                rotation,
                scaling: CONFIG.disp_coef,
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

        let night_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Night Buffer"),
                contents: bytemuck::cast_slice(NIGHT_FILTER_VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );
        let night_instance = vec![Instance {
            position: cgmath::Vector3 { x: 0.0, y: 0.0, z: 0.0 },
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0)),
            scaling: 1.0,
        }];
        let night_instance_data = night_instance.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let night_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&night_instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let map_side_length = 2 * map_render_distance + 1;
        let map_scaling_coef =  2.0 / map_side_length as f32;
        let map_instances =
            (0..map_side_length).flat_map(move |y| {
                (0..map_side_length).map(move |x| {
                    let position =
                        cgmath::Vector3 {
                            x: x as f32 * map_scaling_coef,
                            y: y as f32 * map_scaling_coef,
                            z: 0.0
                        } + INITIAL_POS;

                    Instance {
                        position,
                        rotation: cgmath::Quaternion::from_axis_angle(
                            cgmath::Vector3::unit_z(), cgmath::Deg(0.0)),
                        scaling: map_scaling_coef,
                    }
                })
            }).collect::<Vec<_>>();

        let map_instance_data = map_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let map_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&map_instance_data),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let hp_bar_vertex_buffers = vec![];
        let hp_bar_instance_data: Vec<InstanceRaw> = vec![];
        let hp_bar_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&hp_bar_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
        
        let animation_vertex_buffers: Vec<Buffer> = vec![];
        let projectile_vertex_buffers: Vec<Buffer> = vec![];
        let projectile_instance_data: Vec<InstanceRaw> = vec![];
        let projectile_instance_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&projectile_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        Self {
            vertex_buffer,
            player_vertex_buffer,
            index_buffer,
            instance_buffer,
            player_instance_buffer,
            night_vertex_buffer,
            night_instance_buffer,
            map_instance_buffer,
            hp_bar_vertex_buffers,
            hp_bar_instance_buffer,
            animation_vertex_buffers,
            projectile_vertex_buffers,
            projectile_instance_buffer,
        }
    }
}

pub const INITIAL_POS: cgmath::Vector3<f32> = cgmath::Vector3::new(
    -1.0,
    -1.0,
    0.0,
);
