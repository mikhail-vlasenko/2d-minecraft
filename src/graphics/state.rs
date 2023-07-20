use std::{iter};
use std::time::Instant;

use cgmath::{InnerSpace, Rotation3, Zero};
use lazy_static::lazy_static;
use rand::random;
use strum::IntoEnumIterator;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, include_wgsl, InstanceDescriptor, RenderPass, TextureFormat, TextureView};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    window::Window,
};
use winit::dpi::PhysicalSize;
use crate::crafting::consumable::Consumable;
use crate::crafting::items::Item;

use crate::character::player::Player;
use crate::crafting::interactable::InteractableKind;
use crate::graphics::buffers::Buffers;
use crate::graphics::egui_manager::EguiManager;
use crate::graphics::instance::*;
use crate::graphics::texture_bind_groups::TextureBindGroups;
use crate::graphics::vertex::{HP_BAR_SCALING_COEF, INDICES, make_hp_vertices, PLAYER_VERTICES, Vertex, VERTICES};
use crate::input_decoding::act;
use crate::map_generation::mobs::mob_kind::MobKind;
use crate::map_generation::field::Field;
use crate::crafting::material::Material;
use crate::crafting::ranged_weapon::RangedWeapon;
use crate::crafting::storable::Storable;
use crate::map_generation::chunk::Chunk;
use crate::map_generation::read_chunk::read_file;
use crate::SETTINGS;
use crate::settings::DEFAULT_SETTINGS;

pub const TILES_PER_ROW: u32 = DEFAULT_SETTINGS.window.tiles_per_row as u32;
pub const DISP_COEF: f32 = 2.0 / TILES_PER_ROW as f32;
pub const INITIAL_POS: cgmath::Vector3<f32> = cgmath::Vector3::new(
    -1.0,
    -1.0,
    0.0,
);
pub const RENDER_DISTANCE: usize = ((TILES_PER_ROW - 1) / 2) as usize;
const ROTATION_SPEED: f32 = 2.0 * std::f32::consts::PI / 60.0;

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
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let surface = unsafe { instance.create_surface(window) }.unwrap();
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
            format: surface.get_capabilities(&adapter).formats[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: Default::default(),
            view_formats: vec![],
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

        let clear_color = wgpu::Color { r: 0.1, g: 0.2, b: 0.3, a: 1.0 };

        let mut field = if SETTINGS.field.from_test_chunk {
            let test_chunk = Chunk::from(read_file(String::from("res/chunks/test_chunk.txt")));
            Field::new(RENDER_DISTANCE, Some(test_chunk))
        } else {
            Field::new(RENDER_DISTANCE, None)
        };

        let buffers = Buffers::new(&device, field.get_map_render_distance() as i32);

        let mut player = Player::new(&field);
        player.pickup(Storable::C(Consumable::Apple), 2);
        if SETTINGS.player.cheating_start {
            player.receive_cheat_package();
        }

        // spawn some initial mobs
        let amount = (0.2 * (field.get_loading_distance() * 2 + 1 as usize).pow(2) as f32) as usize;
        field.spawn_mobs(&player, amount, true);
        field.spawn_mobs(&player, amount * 2, false);

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
                input: KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode,
                    ..
                },
                ..
            } => {
                if self.player.get_hp() > 0 {
                    self.player.message = String::new();
                    // different actions take different time, so sometimes mobs are not allowed to step
                    let passed_time = act(virtual_keycode,
                                          &mut self.player, &mut self.field,
                                          &self.egui_manager.craft_menu_open);
                    self.field.step_time(passed_time, &mut self.player);
                }
                true
            }
            _ => false,
        }
    }

    pub fn update(&mut self) {
        let mob_positions = self.field.all_mob_positions_and_hp(&self.player);

        self.buffers.hp_bar_vertex_buffer = vec![];
        for m in &mob_positions {
            self.buffers.hp_bar_vertex_buffer.push(self.hp_bar_vertices(1.));
            self.buffers.hp_bar_vertex_buffer.push(self.hp_bar_vertices(m.2));
        }
        self.buffers.hp_bar_instances = vec![];
        for m in &mob_positions {
            self.buffers.hp_bar_instances.push(self.hp_bar_position_instance(m.0, m.1));
            self.buffers.hp_bar_instances.push(self.hp_bar_position_instance(m.0, m.1));
        }
        let hp_bar_instance_data = self.buffers.hp_bar_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        self.buffers.hp_bar_instance_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&hp_bar_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
    }

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

        if !self.player.viewing_map {
            self.render_game(&mut render_pass);
        } else {
            self.render_map(&mut render_pass);
        }
        drop(render_pass);

        let texture_delta = self.egui_manager.render_ui(
            &self.config, &self.device, &self.queue, &mut encoder, &view, window,
            &mut self.player, &mut self.field,
        );

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        self.egui_manager.post_render_cleanup(texture_delta);

        Ok(())
    }

    fn render_game<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        self.render_field(render_pass);
        self.render_mobs(render_pass);

        // render player
        render_pass.set_vertex_buffer(0, self.buffers.player_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.player_instance_buffer.slice(..));
        render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_player(), &[]);
        let idx = self.player.get_rotation();
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx + 1);

        self.render_hp_bars(render_pass);

        // render night filter
        self.render_night(render_pass);
    }

    pub fn handle_ui_event<T>(&mut self, event: &Event<T>) {
        self.egui_manager.handle_event(&event);
    }

    /// Converts coordinates and rotation to an index in the buffer
    ///
    /// # Arguments
    ///
    /// * `x` - x coordinate (as usual, the vertical one)
    /// * `y` - y coordinate (as usual, the horizontal one)
    /// * `rotation` - rotation of the texture (number from 0 to 3)
    fn convert_index(x: i32, y: i32, rotation: u32) -> u32 {
        // Here we want x to be horizontal, like mathematical coords
        // Also, second component should be greater when higher (so negate it)
        (-x + RENDER_DISTANCE as i32) as u32 * TILES_PER_ROW
            + (y + RENDER_DISTANCE as i32) as u32
            + rotation * TILES_PER_ROW.pow(2)
    }

    fn hp_bar_position_instance(&self, mob_x: i32, mob_y: i32) -> Instance {
        Instance {
            position: cgmath::Vector3 {
                x: (mob_y as f32 - 0.5) * DISP_COEF,
                y: (-mob_x as f32 + 0.3) * DISP_COEF,
                z: 0.0
            },
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0)),
            scaling: HP_BAR_SCALING_COEF,
        }
    }

    fn hp_bar_vertices(&self, hp_share: f32) -> Buffer {
        self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("HP Bar Vertex Buffer"),
                contents: bytemuck::cast_slice(&make_hp_vertices(hp_share)),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        )
    }

    fn draw_at_indices(&self, indices: &Vec<(i32, i32)>, render_pass: &mut RenderPass, rotations: Option<Vec<u32>>) {
        let rots = if rotations.is_some() {
            rotations.unwrap()
        } else {
            vec![0; indices.len()]
        };
        for i in 0..indices.len() {
            let pos = indices[i];
            let idx = State::convert_index(pos.0, pos.1, rots[i]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx + 1);
        }
    }

    /// Draws top material or texture. in case of texture also draws material underneath
    fn draw_material<'a>(&'a self, i: i32, j: i32, render_pass: &mut RenderPass<'a>, map: bool) {
        let material = self.field.top_material_at((i, j));
        let idx = if map {
            self.convert_map_view_index(i - self.player.x, j - self.player.y)
        } else {
            State::convert_index(i - self.player.x, j - self.player.y, 0)
        };
        if let Material::Texture(_) = material {
            let non_texture = self.field.non_texture_material_at((i, j));
            render_pass.set_bind_group(
                0, self.bind_groups.get_bind_group_material(non_texture), &[]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx + 1);
        }
        render_pass.set_bind_group(
            0, self.bind_groups.get_bind_group_material(material), &[]);
        render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx + 1);
    }

    /// Draws textures of top materials on every tile, then draws depth indicators on top.
    /// Also draws other key components (interactables, loot).
    ///
    /// # Arguments
    ///
    /// * `render_pass`: the primary render pass
    fn render_field<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        // let now = Instant::now();
        // draw materials of top block in tiles
        for i in (self.player.x - RENDER_DISTANCE as i32)..=(self.player.x + RENDER_DISTANCE as i32) {
            for j in (self.player.y - RENDER_DISTANCE as i32)..=(self.player.y + RENDER_DISTANCE as i32) {
                self.draw_material(i, j, render_pass, false);
            }
        }

        // draw depth indicators on top of the tiles
        for i in 0..=3 {
            render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_depth(i), &[]);
            let depth = self.field.depth_indices(&self.player, i + 2);
            self.draw_at_indices(&depth, &mut *render_pass, None);
        }

        // draw interactable objects
        for interactable in InteractableKind::iter() {
            render_pass.set_bind_group(0,
                                       &self.bind_groups.get_bind_group_interactable(interactable),
                                       &[]);
            let interactables = self.field.interactable_indices(&self.player, interactable);
            self.draw_at_indices(&interactables, &mut *render_pass, None);
        }

        // draw loot where exists
        render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_loot(), &[]);
        let loot = self.field.loot_indices(&self.player);
        self.draw_at_indices(&loot, &mut *render_pass, None);

        // draw arrows left from shooting
        render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_arrow(), &[]);
        let loot = self.field.arrow_indices(&self.player);
        self.draw_at_indices(&loot, &mut *render_pass, None);
        // let elapsed = now.elapsed();
        // println!("Elapsed: {:.2?}", elapsed);
    }

    fn render_mobs<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        let max_drawable_index = ((TILES_PER_ROW - 1) / 2) as i32;
        for mob_kind in MobKind::iter() {
            render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.buffers.instance_buffer.slice(..));
            render_pass.set_bind_group(0, self.bind_groups.get_bind_group_mob(mob_kind), &[]);

            let mut mobs = self.field.mob_indices(&self.player, mob_kind);
            mobs = mobs.into_iter().filter(
                |(x, y, _)| x.abs() <= max_drawable_index && y.abs() <= max_drawable_index
            ).collect();
            let rotations: Vec<u32> = mobs.clone().into_iter().map(|(_, _, rot)| rot).collect();
            let positions = mobs.into_iter().map(|(x, y, _)| (x, y)).collect();
            self.draw_at_indices(&positions, &mut *render_pass, Some(rotations));
        }
    }

    fn render_hp_bars<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_vertex_buffer(1, self.buffers.hp_bar_instance_buffer.slice(..));

        for i in 0..self.buffers.hp_bar_vertex_buffer.len() {
            let red = i % 2 == 0;
            render_pass.set_vertex_buffer(0, self.buffers.hp_bar_vertex_buffer[i].slice(..));
            render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_hp_bar(red), &[]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, i as u32..(i+1) as u32);
        }
    }

    fn render_night<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        if self.field.is_night() {
            if self.field.is_red_moon() {
                render_pass.set_bind_group(0, self.bind_groups.get_bind_group_red_moon(), &[]);
            } else {
                render_pass.set_bind_group(0, self.bind_groups.get_bind_group_night(), &[]);
            }
            render_pass.set_vertex_buffer(0, self.buffers.night_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.buffers.night_instance_buffer.slice(..));
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
    }

    fn convert_map_view_index(&self, x: i32, y: i32) -> u32 {
        (-x + self.field.get_map_render_distance() as i32) as u32 *
            ((self.field.get_map_render_distance() as u32 * 2) + 1) +
            (y + self.field.get_map_render_distance() as i32) as u32
    }

    /// Only renders the materials, but with a much larger render distance.
    fn render_map<'a>(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.map_instance_buffer.slice(..));
        render_pass.set_index_buffer(self.buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        let radius = self.field.get_map_render_distance() as i32;
        for i in (self.player.x - radius)..=(self.player.x + radius) {
            for j in (self.player.y - radius)..=(self.player.y + radius) {
                self.draw_material(i, j, render_pass, true);
            }
        }
    }
}
