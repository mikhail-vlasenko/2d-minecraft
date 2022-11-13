use std::iter;

use cgmath::{InnerSpace, Rotation3, Zero};
use strum::IntoEnumIterator;
use wgpu::{BindGroup, BindGroupLayout, CommandEncoder, include_wgsl, RenderPass, TextureView};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    window::Window,
};
use winit::dpi::PhysicalSize;
use crate::crafting::consumable::Consumable;

use crate::player::Player;
use crate::graphics::buffers::Buffers;
use crate::graphics::egui_manager::EguiManager;
use crate::graphics::instance::*;
use crate::graphics::texture_bind_groups::TextureBindGroups;
use crate::graphics::vertex::{INDICES, PLAYER_VERTICES, Vertex, VERTICES};
use crate::input_decoding::act;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::map_generation::field::Field;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;

pub const TILES_PER_ROW: u32 = 17;
pub const DISP_COEF: f32 = 2.0 / TILES_PER_ROW as f32;
pub const INITIAL_POS: cgmath::Vector3<f32> = cgmath::Vector3::new(
    -1.0,
    -1.0,
    0.0,
);

/// The main class of the application.
/// Initializes graphics.
/// Catches input.
/// Renders the playing grid.
/// Owns the Player and the Field.
pub struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    buffers: Buffers,
    bind_groups: TextureBindGroups,
    egui_manager: EguiManager,
    field: Field,
    player: Player,
    /// when reaches 1, mobs (and other things) are allowed to step
    turn_state: f32,
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

        let egui_manager = EguiManager::new(window, &size, &surface, &adapter, &device);

        let bind_groups = TextureBindGroups::new(&device, &queue);

        let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_groups.bind_group_layout],
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

        let buffers = Buffers::new(&device);

        let clear_color = wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0, };

        // let test_chunk = Chunk::from(read_file(String::from("res/chunks/test_chunk.txt")));
        let field = Field::new(None);
        let mut player = Player::new(&field);
        player.pickup(Storable::C(Consumable::Apple), 2);
        let turn_state = 0.;

        Self {
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            render_pipeline,
            buffers,
            bind_groups,
            egui_manager,
            field,
            player,
            turn_state,
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
                if self.player.get_hp() > 0 {
                    self.player.message = String::new();
                    // different actions take different time, so sometimes mobs are not allowed to step
                    self.turn_state += act(virtual_keycode, &mut self.player, &mut self.field);
                    while self.turn_state >= 1. {
                        self.field.step_mobs(&mut self.player);
                        self.turn_state -= 1.
                    }
                }
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

        let texture_delta = self.egui_manager.render_ui(
            &self.config, &self.device, &self.queue, &mut encoder, &view, window, &mut self.player, self.turn_state
        );

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        self.egui_manager.post_render_cleanup(texture_delta);

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
        render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        self.render_field(&mut render_pass);
        self.render_mobs(&mut render_pass);

        // render player
        render_pass.set_vertex_buffer(0, self.buffers.player_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.player_instance_buffer.slice(..));
        render_pass.set_bind_group(0, &self.bind_groups.player, &[]);
        let idx = self.player.get_rotation();
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx+1);
    }

    pub fn handle_ui_event<T>(&mut self, event: &Event<T>) {
        self.egui_manager.handle_event(&event);
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

    /// Draws textures of top materials on every tile, then draws depth indicators on top.
    ///
    /// # Arguments
    ///
    /// * `render_pass`: the primary render pass
    fn render_field<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        let radius = ((TILES_PER_ROW - 1) / 2) as usize;

        // draw tiles of the same material together
        for material in Material::iter() {
            render_pass.set_bind_group(0, self.bind_groups.get_bind_group_material(material), &[]);
            let tiles = self.field.texture_indices(&self.player, material, radius);
            self.draw_at_indices(tiles, &mut *render_pass);
        }

        // draw depth indicators on top of the tiles
        for i in 0..=3 {
            render_pass.set_bind_group(0, &self.bind_groups.depth_indicators[i], &[]);
            let depth = self.field.depth_indices(&self.player, i+2, radius);
            self.draw_at_indices(depth, &mut *render_pass);
        }
    }

    fn render_mobs<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        let max_drawable_index = ((TILES_PER_ROW - 1) / 2) as i32;
        for mob_kind in MobKind::iter() {
            render_pass.set_bind_group(0, self.bind_groups.get_bind_group_mob(mob_kind), &[]);
            let mut indices = self.field.mob_indices(&self.player, mob_kind);
            indices = indices.into_iter().filter(
                |(x, y)| x.abs() <= max_drawable_index && y.abs() <= max_drawable_index
            ).collect();
            self.draw_at_indices(indices, &mut *render_pass);
        }
    }
}
