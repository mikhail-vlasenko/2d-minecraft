use std::iter;
use cgmath::{InnerSpace, Rotation3, Zero};
use wgpu::{BindGroup, BindGroupLayout, CommandEncoder, include_wgsl, RenderPass, TextureView};
use wgpu::util::DeviceExt;


use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use winit::dpi::PhysicalSize;
use crate::{Field, Player};
use crate::graphics::instance::*;
use crate::graphics::texture::Texture;
use crate::graphics::vertex::{INDICES, PLAYER_VERTICES, Vertex, VERTICES};
use crate::input_decoding::act;
use crate::material::Material;

use ::egui::FontDefinitions;
use egui_wgpu_backend;
use egui_wgpu_backend::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use crate::graphics::texture_bind_groups::TextureBindGroups;


pub const TILES_PER_ROW: u32 = 11;
pub const DISP_COEF: f32 = 2.0 / TILES_PER_ROW as f32;
pub const INITIAL_POS: cgmath::Vector3<f32> = cgmath::Vector3::new(
    -1.0,
    -1.0,
    0.0,
);

pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    player_vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    bind_groups: TextureBindGroups,
    field: Field,
    player: Player,
    egui_platform: Platform,
    egui_rpass: egui_wgpu_backend::RenderPass,
    current_material: Material,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: None,
            },
            None, // Trace path
        ).await.unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let surface_format = surface.get_supported_formats(&adapter)[0];

        let egui_platform = Platform::new(PlatformDescriptor {
            physical_width: size.width as u32,
            physical_height: size.height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });

        let egui_rpass = egui_wgpu_backend::RenderPass::new(&device, surface_format, 1);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let bind_groups = TextureBindGroups::init_bind_groups(&device, &queue, &texture_bind_group_layout);

        let instances = (0..TILES_PER_ROW).flat_map(|y| {
            (0..TILES_PER_ROW).map(move |x| {
                let position =
                    cgmath::Vector3 { x: x as f32 * DISP_COEF, y: y as f32 * DISP_COEF, z: 0.0 }
                        + INITIAL_POS;

                let rotation=
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

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                    InstanceRaw::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1, // 2.
                mask: !0, // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        });

        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
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

        let clear_color = wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0, };

        let player = Player::new(25, 25);
        let field = Field::new();
        field.render(&player);

        let current_material = Material::Dirt;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            bind_groups,
            field,
            player,
            player_vertex_buffer,
            egui_platform,
            egui_rpass,
            current_material,
        }
    }

    pub fn get_size(&self) -> PhysicalSize<u32> {
        self.size
    }



    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.clear_color = wgpu::Color {
                    r: position.x as f64 / self.size.width as f64,
                    g: position.y as f64 / self.size.height as f64,
                    b: 1.0,
                    a: 1.0,
                };
                true
            }
            WindowEvent::KeyboardInput {
                input: KeyboardInput{
                    state: ElementState::Pressed,
                    virtual_keycode,
                    ..
                },
                ..
            } => {
                act(virtual_keycode, &mut self.player, &mut self.field);
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self) { }

    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        self.render_game(&mut encoder, &view);

        let texture_delta = self.render_ui(&mut encoder, &view, window);

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        self.egui_rpass
            .remove_textures(texture_delta)
            .expect("remove texture ok");

        Ok(())
    }

    fn render_game(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear_color),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        self.render_field(&mut render_pass);

        // render player
        render_pass.set_vertex_buffer(0, self.player_vertex_buffer.slice(..));
        render_pass.set_bind_group(0, &self.bind_groups.player, &[]);
        let idx = State::convert_index(0, 0);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx+1);
    }

    fn render_ui(&mut self, encoder: &mut CommandEncoder, view: &TextureView, window: &Window) -> egui::TexturesDelta {
        self.egui_platform.begin_frame();

        egui::Window::new("My Window").show(&self.egui_platform.context(), |ui| {
            ui.label("Placing material");
            ui.radio_value(&mut self.current_material, Material::Dirt, "dirt");
            ui.radio_value(&mut self.current_material, Material::Stone, "stone");
            ui.radio_value(&mut self.current_material, Material::TreeLog, "tree log");
        });

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let full_output = self.egui_platform.end_frame(Some(window));
        let paint_jobs = self.egui_platform.context().tessellate(full_output.shapes);

        // Upload all resources for the GPU.
        let screen_descriptor = ScreenDescriptor {
            physical_width: self.config.width,
            physical_height: self.config.height,
            scale_factor: window.scale_factor() as f32,
        };

        let texture_delta: egui::TexturesDelta = full_output.textures_delta;
        self.egui_rpass
            .add_textures(&self.device, &self.queue, &texture_delta)
            .expect("add texture ok");
        self.egui_rpass.update_buffers(&self.device, &self.queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.egui_rpass
            .execute(
                encoder,
                &view,
                &paint_jobs,
                &screen_descriptor,
                None,
            )
            .unwrap();

        texture_delta
    }

    pub fn handle_ui_event<T>(&mut self, event: &Event<T>) {
        self.egui_platform.handle_event(&event);
    }

    fn convert_index(x: i32, y: i32) -> u32 {
        let radius = (TILES_PER_ROW - 1) / 2;
        (y + radius as i32) as u32 * TILES_PER_ROW + (x + radius as i32) as u32
    }

    fn draw_at_indices(&self, indices: Vec<(i32, i32)>, render_pass: &mut RenderPass) {
        for pos in indices {
            // Here we want x to be horizontal, like mathematical coords
            // Also, second component should be greater when higher (so negate it)
            let idx = State::convert_index(pos.1, -pos.0);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx+1);
        }
    }

    fn render_field<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        let radius = ((TILES_PER_ROW - 1) / 2) as usize;

        render_pass.set_bind_group(0, &self.bind_groups.grass, &[]);
        let grass = self.field.texture_indices(&self.player, Material::Dirt, radius);
        self.draw_at_indices(grass, &mut *render_pass);

        render_pass.set_bind_group(0, &self.bind_groups.stone, &[]);
        let stone = self.field.texture_indices(&self.player, Material::Stone, radius);
        self.draw_at_indices(stone, &mut *render_pass);

        render_pass.set_bind_group(0, &self.bind_groups.tree_log, &[]);
        let tree = self.field.texture_indices(&self.player, Material::TreeLog, radius);
        self.draw_at_indices(tree, &mut *render_pass);

        render_pass.set_bind_group(0, &self.bind_groups.bedrock, &[]);
        let bedrock = self.field.texture_indices(&self.player, Material::Bedrock, radius);
        self.draw_at_indices(bedrock, &mut *render_pass);
    }
}
