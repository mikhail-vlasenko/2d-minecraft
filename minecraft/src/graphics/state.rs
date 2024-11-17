use std::{iter};
use std::path::{PathBuf};

use cgmath::{Rotation3};
use strum::IntoEnumIterator;
use egui_wgpu::wgpu;
use egui_wgpu::wgpu::{Buffer, include_wgsl, InstanceDescriptor, RenderPass, StoreOp};
use egui_wgpu::wgpu::util::DeviceExt;
use egui_winit::winit::{
    event::*,
    window::Window,
};
use egui_winit::winit::dpi::PhysicalSize;
use egui_winit::winit::keyboard::{KeyCode, PhysicalKey};
use game_logic::crafting::consumable::Consumable;

use game_logic::character::player::Player;
use game_logic::crafting::interactable::InteractableKind;
use crate::graphics::buffers::Buffers;
use crate::graphics::ui::egui_manager::EguiManager;
use crate::graphics::instance::*;
use crate::graphics::texture_bind_groups::TextureBindGroups;
use crate::graphics::vertex::{CENTERED_SQUARE_VERTICES, HP_BAR_SCALING_COEF, INDICES, make_animation_vertices, make_hp_vertices, PROJECTILE_ARROW_VERTICES, Vertex, VERTICES};
use game_logic::perform_action::act;
use game_logic::map_generation::mobs::mob_kind::MobKind;
use game_logic::map_generation::field::{absolute_to_relative, AbsolutePos, Field, RelativePos};
use game_logic::crafting::material::Material;
use game_logic::crafting::storable::Storable;
use game_logic::auxiliary::animations::{AnimationManager, ProjectileType};
use crate::graphics::ui::egui_renderer::EguiRenderer;
use crate::graphics::ui::main_menu::{SecondPanelState, SelectedOption};
use game_logic::map_generation::chunk::Chunk;
use game_logic::map_generation::read_chunk::read_file;
use game_logic::map_generation::save_load::{load_game, save_game};
use game_logic::{handle_action, init_game, SETTINGS};
use game_logic::auxiliary::replay::Replay;
use game_logic::character::milestones::MilestoneTracker;
use crate::graphical_config::CONFIG;
use crate::input_decoding::map_key_to_action;


/// The main class of the application.
/// Initializes graphics.
/// Catches input.
/// Renders the playing grid.
/// Owns the Player and the Field.
pub struct State<'a> {
    surface: wgpu::Surface<'a>,
    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    pub window: &'a Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    buffers: Buffers,
    bind_groups: TextureBindGroups,
    pub egui_renderer: EguiRenderer,
    egui_manager: EguiManager,
    animation_manager: AnimationManager,
    recorded_replay: Replay,
    active_replay: Option<Replay>,
    milestone_tracker: MilestoneTracker,
    field: Field,
    player: Player,
}

impl<'a> State<'a> {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
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
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

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
        
        let egui_renderer = EguiRenderer::new(
            &device,       // wgpu Device
            config.format, // TextureFormat
            None,          // this can be None
            1,             // samples
            window,        // winit Window
        );

        let egui_manager = EguiManager::new();

        let animation_manager = AnimationManager::new();
        
        let active_replay = None;

        let (field, player, recorded_replay, milestone_tracker) = init_game();

        let buffers = Buffers::new(&device, field.get_map_render_distance() as i32);

        Self {
            surface,
            window,
            device,
            queue,
            config,
            size,
            render_pipeline,
            buffers,
            bind_groups,
            egui_renderer,
            egui_manager,
            animation_manager,
            recorded_replay,
            active_replay,
            milestone_tracker,
            field,
            player,
        }
    }

    pub fn new_game(&mut self) {
        let (field, player, replay, milestone_tracker) = init_game();
        self.field = field;
        self.player = player;
        self.recorded_replay = replay;
        self.milestone_tracker = milestone_tracker;
    }

    pub fn get_size(&self) -> PhysicalSize<u32> {
        self.size
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn handle_action(&mut self, virtual_keycode: &KeyCode) {
        match virtual_keycode { 
            KeyCode::F1 => {
                if self.active_replay.is_some() {
                    let replay = self.active_replay.as_mut().unwrap();
                    replay.step_back(&mut self.field, &mut self.player)
                }
            }
            KeyCode::F2 => { 
                if self.active_replay.is_some() {
                    let replay = self.active_replay.as_mut().unwrap();
                    if !replay.finished() {
                        replay.apply_state(&mut self.field, &mut self.player);
                    }
                }
            }
            _ => {
                let action = map_key_to_action(virtual_keycode, &self.player);
                if action.is_some() {
                    handle_action(
                        &action.unwrap(),
                        &mut self.field, &mut self.player,
                        &self.egui_manager.main_menu_open,
                        &self.egui_manager.craft_menu_open,
                        Some(&mut self.animation_manager),
                        &mut self.recorded_replay,
                        &mut self.milestone_tracker,
                    );
                }
            }
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        self.window().request_redraw();
        match event {
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    state: ElementState::Pressed,
                    physical_key: PhysicalKey::Code(virtual_keycode),
                    ..
                },
                ..
            } => {
                self.handle_action(virtual_keycode);
                false
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                match button {
                    // the 2 mouse buttons are mapped to keyboard keys to allow only one hand on the keyboard
                    MouseButton::Forward => {
                        self.handle_action(&KeyCode::ArrowRight);
                    }
                    MouseButton::Back => {
                        self.handle_action(&KeyCode::ArrowLeft);
                    }
                    _ => {}
                }
                false
            }
            _ => false,
        }
    }

    pub fn update(&mut self) {
        if self.egui_manager.main_menu.selected_option == SelectedOption::NewGame {
            self.new_game();
            self.egui_manager.main_menu_open.replace(false);
            self.egui_manager.main_menu.selected_option = SelectedOption::Nothing;
        }

        if self.egui_manager.main_menu.selected_option == SelectedOption::SaveGame {
            let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
            path.push(self.egui_manager.main_menu.save_name.clone());
            save_game(&self.field, &self.player, &self.milestone_tracker, path.as_path());
            self.egui_manager.main_menu.selected_option = SelectedOption::Nothing;
            self.egui_manager.main_menu.second_panel = SecondPanelState::About;
        }

        if self.egui_manager.main_menu.selected_option == SelectedOption::LoadGame {
            let mut path = PathBuf::from(SETTINGS.read().unwrap().save_folder.clone().into_owned());
            path.push(self.egui_manager.main_menu.save_name.clone());
            let (field, player, milestone_tracker) = load_game(path.as_path());
            self.field = field;
            self.player = player;
            self.milestone_tracker = milestone_tracker;
            self.egui_manager.main_menu_open.replace(false);
            self.egui_manager.main_menu.selected_option = SelectedOption::Nothing;
            self.egui_manager.main_menu.second_panel = SecondPanelState::About;
            self.animation_manager.clear();
        }
        
        if self.egui_manager.main_menu.selected_option == SelectedOption::WatchReplay {
            let mut path = PathBuf::from(SETTINGS.read().unwrap().replay_folder.clone().into_owned());
            path.push(self.egui_manager.main_menu.replay_name.clone());
            self.active_replay = Some(Replay::load(path.as_path()));
            self.egui_manager.main_menu_open.replace(false);
            self.egui_manager.main_menu.selected_option = SelectedOption::Nothing;
            self.egui_manager.main_menu.second_panel = SecondPanelState::About;
            self.animation_manager.clear();
            // start the replay
            self.handle_action(&KeyCode::F2);
        }
        
        if self.egui_manager.save_replay_clicked {
            let path = PathBuf::from(SETTINGS.read().unwrap().replay_folder.clone().into_owned());
            let name = self.recorded_replay.make_save_name();
            let path = path.join(name.clone());
            self.recorded_replay.save(path.as_path());
            self.egui_manager.save_replay_clicked = false;
            self.egui_manager.replay_save_name = Some(name);
        }

        let mob_positions_and_hp = self.field.close_mob_info(|mob| {
                let pos = absolute_to_relative((mob.pos.x, mob.pos.y), &self.player);
                (pos, mob.get_hp_share())
            }, &self.player);

        self.buffers.hp_bar_vertex_buffers = vec![];
        for mob_info in &mob_positions_and_hp {
            self.buffers.hp_bar_vertex_buffers.push(self.hp_bar_vertices(1.));
            self.buffers.hp_bar_vertex_buffers.push(self.hp_bar_vertices(mob_info.1));
        }
        let mut hp_bar_instances = vec![];
        for mob_info in &mob_positions_and_hp {
            hp_bar_instances.push(self.hp_bar_position_instance(mob_info.0));
            hp_bar_instances.push(self.hp_bar_position_instance(mob_info.0));
        }
        let hp_bar_instance_data = hp_bar_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        self.buffers.hp_bar_instance_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&hp_bar_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );

        self.update_animations();
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.5, g: 0.7, b: 0.5, a: 1.0 }),
                        store: StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            render_pass.set_pipeline(&self.render_pipeline);

            if !self.player.viewing_map {
                self.render_game(&mut render_pass);
            } else {
                self.render_map(&mut render_pass);
            }
        }

        self.egui_manager.render_ui(
            &mut self.egui_renderer,
            &self.config, &self.device, &self.queue, &mut encoder, &view, self.window,
            &mut self.player, &mut self.field,
        );

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn render_game(&'a self, render_pass: &mut RenderPass<'a>) {
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
        
        self.render_animations(render_pass);

        self.render_hp_bars(render_pass);

        // render night filter
        self.render_night(render_pass);
    }

    /// Converts coordinates and rotation to an index in the buffer
    ///
    /// # Arguments
    ///
    /// * `pos` - relative position with the first element as x coordinate (as usual, the vertical one)
    ///     and second element as y coordinate (as usual, the horizontal one)
    /// * `rotation` - rotation of the texture (number from 0 to 3)
    fn convert_index(pos: RelativePos, rotation: u32) -> u32 {
        // Here we want x to be horizontal, like mathematical coords
        // Also, second component should be greater when higher (so negate it)
        (-pos.0 + CONFIG.render_distance as i32) as u32 * CONFIG.tiles_per_row
            + (pos.1 + CONFIG.render_distance as i32) as u32
            + rotation * CONFIG.tiles_per_row.pow(2)
    }

    fn hp_bar_position_instance(&self, mob_pos: RelativePos) -> Instance {
        Instance {
            position: cgmath::Vector3 {
                x: (mob_pos.1 as f32 - 0.5) * CONFIG.disp_coef,
                y: (-mob_pos.0 as f32 + 0.3) * CONFIG.disp_coef,
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
    
    /// position is relative, but float
    fn projectile_instance(&self, position: (f32, f32), rotation: f32) -> Instance {
        Instance {
            position: cgmath::Vector3 {
                x: (position.1 - 0.5) * CONFIG.disp_coef,
                y: (-position.0 + 0.5) * CONFIG.disp_coef,
                z: 0.0
            },
            rotation: cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Rad(rotation)),
            scaling: CONFIG.disp_coef,
        }
    }

    /// Draws the current texture at the player-centered grid positions.
    fn draw_at_grid_positions(&self, positions: &Vec<RelativePos>, render_pass: &mut RenderPass, rotations: Option<Vec<u32>>) {
        let rots = if rotations.is_some() {
            rotations.unwrap()
        } else {
            vec![0; positions.len()]
        };
        for i in 0..positions.len() {
            let pos = positions[i];
            let idx = State::convert_index(pos, rots[i]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, idx..idx + 1);
        }
    }

    /// Draws top material or texture. in case of texture also draws material underneath
    fn draw_material(&'a self, pos: AbsolutePos, render_pass: &mut RenderPass<'a>, map: bool) {
        let material = self.field.top_material_at(pos);
        let idx = if map {
            self.convert_map_view_index(pos.0 - self.player.x, pos.1 - self.player.y)
        } else {
            State::convert_index((pos.0 - self.player.x, pos.1 - self.player.y), 0)
        };
        if let Material::Texture(_) = material {
            let non_texture = self.field.non_texture_material_at(pos);
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
    fn render_field(&'a self, render_pass: &mut RenderPass<'a>) {
        // let now = Instant::now();
        // draw materials of top block in tiles
        for i in (self.player.x - CONFIG.render_distance as i32)..=(self.player.x + CONFIG.render_distance as i32) {
            for j in (self.player.y - CONFIG.render_distance as i32)..=(self.player.y + CONFIG.render_distance as i32) {
                self.draw_material((i, j), render_pass, false);
            }
        }

        // draw depth indicators on top of the tiles
        for i in 0..=3 {
            render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_depth(i), &[]);
            let depth = self.field.depth_indices(&self.player, i + 2);
            self.draw_at_grid_positions(&depth, &mut *render_pass, None);
        }

        // draw interactable objects
        for interactable in InteractableKind::iter() {
            render_pass.set_bind_group(0,
                                       &self.bind_groups.get_bind_group_interactable(interactable),
                                       &[]);
            let interactables = self.field.interactable_indices(&self.player, interactable);
            self.draw_at_grid_positions(&interactables, &mut *render_pass, None);
        }

        // draw loot where exists
        render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_loot(), &[]);
        let loot = self.field.loot_indices(&self.player);
        self.draw_at_grid_positions(&loot, &mut *render_pass, None);

        // draw arrows left from shooting
        render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_arrow(), &[]);
        let loot = self.field.arrow_indices(&self.player);
        self.draw_at_grid_positions(&loot, &mut *render_pass, None);
        // let elapsed = now.elapsed();
        // println!("Elapsed: {:.2?}", elapsed);
    }

    fn render_mobs(&'a self, render_pass: &mut RenderPass<'a>) {
        let max_drawable_index = ((CONFIG.tiles_per_row - 1) / 2) as i32;
        for mob_kind in MobKind::iter() {
            render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.buffers.instance_buffer.slice(..));
            render_pass.set_bind_group(0, self.bind_groups.get_bind_group_mob(mob_kind), &[]);

            let mut mobs = self.field.mob_indices(&self.player, mob_kind);
            mobs = mobs.into_iter().filter(
                |(pos, _)| pos.0.abs() <= max_drawable_index && pos.1.abs() <= max_drawable_index
            ).collect();
            let rotations: Vec<u32> = mobs.clone().into_iter().map(|(_, rot)| rot).collect();
            let positions = mobs.into_iter().map(|(pos, _)| pos).collect();
            self.draw_at_grid_positions(&positions, &mut *render_pass, Some(rotations));
        }
    }

    fn render_hp_bars(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_vertex_buffer(1, self.buffers.hp_bar_instance_buffer.slice(..));

        for i in 0..self.buffers.hp_bar_vertex_buffers.len() {
            let red = i % 2 == 0;
            render_pass.set_vertex_buffer(0, self.buffers.hp_bar_vertex_buffers[i].slice(..));
            render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_hp_bar(red), &[]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, i as u32..(i+1) as u32);
        }
    }

    fn render_animations(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_vertex_buffer(1, self.buffers.instance_buffer.slice(..));
        for i in 0..self.buffers.animation_vertex_buffers.len() {
            let animation = self.animation_manager.get_tile_animations()[i];
            render_pass.set_vertex_buffer(0, self.buffers.animation_vertex_buffers[i].slice(..));
            render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_animation(*animation.get_animation_type()), &[]);
            let mut rel_animation_positions = vec![absolute_to_relative(*animation.get_pos(), &self.player)];
            rel_animation_positions = self.filter_out_of_view_tiles(rel_animation_positions);
            self.draw_at_grid_positions(&rel_animation_positions, render_pass, None);
        }

        render_pass.set_vertex_buffer(1, self.buffers.projectile_instance_buffer.slice(..));
        for i in 0..self.buffers.projectile_vertex_buffers.len() {
            let animation = self.animation_manager.get_projectile_animations()[i];
            render_pass.set_vertex_buffer(0, self.buffers.projectile_vertex_buffers[i].slice(..));
            render_pass.set_bind_group(0, &self.bind_groups.get_bind_group_projectile(*animation.get_projectile_type()), &[]);
            render_pass.draw_indexed(0..INDICES.len() as u32, 0, i as u32..(i+1) as u32);
        }
    }

    fn render_night(&'a self, render_pass: &mut RenderPass<'a>) {
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
    
    fn update_animations(&mut self) {
        // update before writing vertices. new animations will not change
        self.animation_manager.update();
        // tile animations are animated by moving texture coordinates, so a separate vertex buffer is needed per animation
        self.buffers.animation_vertex_buffers = vec![];
        for tile_animation in self.animation_manager.get_tile_animations() {
            self.buffers.animation_vertex_buffers.push(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Animation Vertex Buffer"),
                    contents: bytemuck::cast_slice(
                        &make_animation_vertices(tile_animation.get_frame(),
                                                 tile_animation.get_num_frames())),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }
            ));
        }
        // projectile animations are animated by moving the projectile position, not strictly over grid tiles,
        // so it needs an instance per animation,
        // and also a separate vertex buffer per animation as projectile animations come in different shapes
        self.buffers.projectile_vertex_buffers = vec![];
        for projectile_animation in self.animation_manager.get_projectile_animations() {
            let vertices = match projectile_animation.get_projectile_type() {
                ProjectileType::Arrow => PROJECTILE_ARROW_VERTICES,
                ProjectileType::GelatinousCube => CENTERED_SQUARE_VERTICES,
            };
            self.buffers.projectile_vertex_buffers.push(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Projectile Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }
            ));
        }
        let mut projectile_instances = vec![];
        for projectile_animation in self.animation_manager.get_projectile_animations() {
            projectile_instances.push(self.projectile_instance(
                projectile_animation.get_relative_position(&self.player), projectile_animation.get_rotation()));
        }
        let projectile_instance_data = projectile_instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        self.buffers.projectile_instance_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(&projectile_instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }
        );
    }

    fn convert_map_view_index(&self, x: i32, y: i32) -> u32 {
        (-x + self.field.get_map_render_distance() as i32) as u32 *
            ((self.field.get_map_render_distance() as u32 * 2) + 1) +
            (y + self.field.get_map_render_distance() as i32) as u32
    }
    
    fn filter_out_of_view_tiles(&self, positions: Vec<RelativePos>) -> Vec<RelativePos> {
        let max_drawable_index = ((CONFIG.tiles_per_row - 1) / 2) as i32;
        positions.into_iter().filter(
            |pos| pos.0.abs() <= max_drawable_index && pos.1.abs() <= max_drawable_index
        ).collect()
    }

    /// Only renders the materials, but with a much larger render distance.
    fn render_map(&'a self, render_pass: &mut RenderPass<'a>) {
        render_pass.set_vertex_buffer(0, self.buffers.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.buffers.map_instance_buffer.slice(..));
        render_pass.set_index_buffer(self.buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        let radius = self.field.get_map_render_distance() as i32;
        for i in (self.player.x - radius)..=(self.player.x + radius) {
            for j in (self.player.y - radius)..=(self.player.y + radius) {
                self.draw_material((i, j), render_pass, true);
            }
        }
    }
}
