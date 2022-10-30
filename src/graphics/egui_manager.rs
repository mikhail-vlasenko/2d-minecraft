use ::egui::FontDefinitions;
use egui::{Align, Align2, Color32, FontId, RichText, TexturesDelta};
use egui_wgpu_backend;
use egui_wgpu_backend::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::{Adapter, CommandEncoder, Device, Queue, Surface, SurfaceConfiguration, TextureView};
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::window::Window;
use crate::{Player};
use strum::IntoEnumIterator;
use crate::crafting::items::Item;
use crate::crafting::material::Material;
use crate::crafting::storable::Storable;


/// Renders UI
pub struct EguiManager {
    platform: Platform,
    render_pass: egui_wgpu_backend::RenderPass,
}

impl EguiManager {
    pub fn new(
        window: &Window,
        size: &PhysicalSize<u32>,
        surface: &Surface,
        adapter: &Adapter,
        device: &Device,
    ) -> Self {
        let surface_format = surface.get_supported_formats(&adapter)[0];

        let platform = Platform::new(PlatformDescriptor {
            physical_width: size.width as u32,
            physical_height: size.height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });

        let render_pass = egui_wgpu_backend::RenderPass::new(&device, surface_format, 1);

        Self {
            platform,
            render_pass,
        }
    }

    pub fn render_ui(&mut self,
                     config: &SurfaceConfiguration,
                     device: &Device,
                     queue: &Queue,
                     encoder: &mut CommandEncoder,
                     view: &TextureView,
                     window: &Window,
                     player: &mut Player,
                     turn_state: f32,
    ) -> TexturesDelta {
        self.platform.begin_frame();

        self.render_place_craft_menu(player);
        self.render_inventory(player);
        self.render_info(player, turn_state);

        if player.get_hp() <= 0 {
            self.render_game_over();
        }

        // End the UI frame. We could now handle the output and draw the UI with the backend.
        let full_output = self.platform.end_frame(Some(window));
        let paint_jobs = self.platform.context().tessellate(full_output.shapes);

        // Upload all resources for the GPU.
        let screen_descriptor = ScreenDescriptor {
            physical_width: config.width,
            physical_height: config.height,
            scale_factor: window.scale_factor() as f32,
        };

        let texture_delta: TexturesDelta = full_output.textures_delta;
        self.render_pass
            .add_textures(&device, &queue, &texture_delta)
            .expect("add texture ok");
        self.render_pass.update_buffers(&device, &queue, &paint_jobs, &screen_descriptor);

        // Record all render passes.
        self.render_pass
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

    pub fn post_render_cleanup(&mut self, texture_delta: TexturesDelta) {
        self.render_pass
            .remove_textures(texture_delta)
            .expect("remove texture ok");
    }

    fn render_place_craft_menu(&self, player: &mut Player) {
        egui::Window::new("Menu").show(&self.platform.context(), |ui| {
            ui.label("Placing material");
            for material in Material::iter() {
                ui.radio_value(&mut player.placement_material, material, material.to_string());
            }

            ui.label("Crafting item");
            for material in Material::iter() {
                if material.craft_yield() > 0 {
                    ui.radio_value(
                        &mut player.crafting_item,
                        Storable::M(material),
                        format!("{} x{}", material.to_string(), material.craft_yield()),
                    );
                }
            }
            for item in Item::iter() {
                if item.craft_yield() > 0 {
                    ui.radio_value(
                        &mut player.crafting_item,
                        Storable::I(item),
                        format!("{} x{}", item.to_string(), item.craft_yield()),
                    );
                }
            }
        });
    }

    fn render_inventory(&self, player: &Player) {
        egui::Window::new("Inventory").show(&self.platform.context(), |ui| {
            for item in player.get_inventory() {
                if item.1 != 0 {
                    ui.label(format!("{}: {}", item.0, item.1));
                }
            }
        });
    }

    fn render_info(&self, player: &Player, turn_state: f32) {
        egui::Window::new("Info").anchor(Align2::RIGHT_TOP, [0., 0.])
            .show(&self.platform.context(), |ui| {
                ui.label(format!("Position: {}, {}, {}", player.x, player.y, player.z));
                ui.label(format!("HP: {}", player.get_hp()));
                ui.label(format!("ATK: {}", player.get_melee_damage()));
                ui.label(format!("Mining PWR: {}", player.get_mining_power()));
                ui.label(format!("Turn state: {}", turn_state));
                ui.label(format!("{}", player.message));
            });
    }

    fn render_game_over(&self) {
        egui::Window::new("").anchor(Align2::CENTER_CENTER, [0., 0.])
            .show(&self.platform.context(), |ui| {
                ui.label(RichText::new("Game Over!")
                    .font(FontId::proportional(80.0))
                    .color(Color32::RED)
                );
            });
    }

    pub fn handle_event<T>(&mut self, event: &Event<T>) {
        self.platform.handle_event(event);
    }
}