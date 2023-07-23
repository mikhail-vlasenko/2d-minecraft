use std::process::exit;
use egui::{Align, Checkbox, FontDefinitions, Slider};
use egui::{Align2, Color32, FontId, Label, RichText, TexturesDelta};
use egui_wgpu_backend;
use egui_wgpu_backend::ScreenDescriptor;
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::{Adapter, CommandEncoder, Device, Queue, Surface, SurfaceConfiguration, TextureView};
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::window::Window;
use crate::character::player::Player;

pub struct MainMenu {
    pub selected_option: u32,
}

impl MainMenu {
    pub fn new() -> Self {
        MainMenu {
            selected_option: 0,
        }
    }

    pub fn render_main_menu(
        &mut self,
        platform: &Platform,
        player: &mut Player,
        width: f32,
    ) {
        egui::Window::new("Main Menu")
            .anchor(Align2::CENTER_CENTER, [0., 0.])
            .collapsible(false)
            .fixed_size([width, 500.])
            .show(&platform.context(), |ui| {
                ui.columns(2, |columns| {
                    columns[0].add_space(50.0); // Add space below buttons

                    columns[0].vertical_centered(|ui| {
                        if ui.button(RichText::new("Start New Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.selected_option = 1;
                        }
                        if ui.button(RichText::new("Load Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.selected_option = 2;
                        }
                        if ui.button(RichText::new("Settings")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            self.selected_option = 3;
                        }
                        if ui.button(RichText::new("Exit Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            exit(0);
                        }
                    });

                    columns[0].add_space(50.0); // Add space below buttons

                    columns[1].label("A short introduction to the game:");
                    columns[1].label("This is a survival game, that take inspiration from Minecraft and Rogue.");
                    columns[1].label("You will have to craft items, fight enemies and explore the world.");
                    columns[1].label("You will be able to build your own base and defend it from enemies.");
                    columns[1].label("Your attack damage depends on the items in your inventory.");
                    columns[1].label("Better pickaxes will allow you to mine more types of blocks.");
                    columns[1].label("Good luck!");
                });
            });
    }
}
