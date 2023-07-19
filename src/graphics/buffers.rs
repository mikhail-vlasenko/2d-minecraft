use cgmath::Rotation3;
use wgpu::Device;
use wgpu::util::DeviceExt;
use crate::graphics::instance::Instance;
use crate::graphics::state::{DISP_COEF, INITIAL_POS, TILES_PER_ROW};
use crate::graphics::vertex::{COLOR_VERTICES, INDICES, NIGHT_FILTER_VERTICES, PLAYER_VERTICES, VERTICES};
use crate::SETTINGS;


/// Creates and stores wgpu buffers
pub struct Buffers {
    pub vertex_buffer: wgpu::Buffer,
    pub color_vertex_buffer: wgpu::Buffer,
    pub player_vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub player_instances: Vec<Instance>,
    pub player_instance_buffer: wgpu::Buffer,
    pub night_vertex_buffer: wgpu::Buffer,
    pub night_instance_buffer: wgpu::Buffer,
    pub map_instance_buffer: wgpu::Buffer,
}

impl Buffers {
    pub fn new(device: &Device, map_render_distance: i32) -> Self {
        let instances = (0..4).flat_map(|angle| {
            let x_rot_compensation = if angle == 1 || angle == 2 { 1 } else { 0 };
            let y_rot_compensation = if angle > 1 { 1 } else { 0 };
            (0..TILES_PER_ROW).flat_map(move |y| {
                (0..TILES_PER_ROW).map(move |x| {
                    let position =
                        cgmath::Vector3 {
                            x: (x + x_rot_compensation) as f32 * DISP_COEF,
                            y: (y + y_rot_compensation) as f32 * DISP_COEF,
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

        let color_vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("hp bar buffer"),
                contents: bytemuck::cast_slice(COLOR_VERTICES),
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
                    z: 0.0,
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
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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

        Self {
            vertex_buffer,
            color_vertex_buffer,
            player_vertex_buffer,
            index_buffer,
            instance_buffer,
            player_instances,
            player_instance_buffer,
            night_vertex_buffer,
            night_instance_buffer,
            map_instance_buffer,
        }
    }
}