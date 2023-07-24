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
use crate::SETTINGS;
use crate::settings::Settings;

pub struct MainMenu {
    pub selected_option: u32,
    pub settings_menu_open: bool,
}

impl MainMenu {
    pub fn new() -> Self {
        MainMenu {
            selected_option: 0,
            settings_menu_open: false,
        }
    }

    pub fn render_main_menu(
        &mut self,
        platform: &Platform,
        player: &mut Player,
        width: f32,
    ) {
        let mut settings = SETTINGS.write().unwrap();  // lock SETTINGS once at the beginning
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
                            self.settings_menu_open = true;
                        }
                        if ui.button(RichText::new("Exit Game")
                            .font(FontId::proportional(30.0))
                            .strong()
                        ).clicked() {
                            exit(0);
                        }
                    });

                    columns[0].add_space(50.0); // Add space below buttons

                    if self.settings_menu_open {
                        columns[1].label(RichText::new("Settings")
                            .font(FontId::proportional(20.0))
                        );

                        Self::render_difficulty_buttons(&mut columns[1], &mut settings);

                        columns[1].add_space(10.0);

                        columns[1].label("Mobs Spawning:");
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.initial_hostile_per_chunk, 0.0..=1.0)
                            .text("Initial Hostile per Chunk"));
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.base_day_amount, 0..=10)
                            .text("Base Day Amount"));
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.base_night_amount, 0..=10)
                            .text("Base Night Amount"));
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.increase_amount_every, 1..=10)
                            .text("Increase Amount Every (days)"));
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.max_mobs_on_chunk, 0..=10)
                            .text("Max Mobs on Chunk"));
                        columns[1].label("Hostile Mob Kind Probabilities (otherwise zombie):");
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.probabilities.bane, 0.0..=1.0)
                            .text("Baneling"));
                        columns[1].add(Slider::new(&mut settings.mobs.spawning.probabilities.ling, 0.0..=1.0)
                            .text("Zerging"));

                        columns[1].label("Field Generation:");
                        columns[1].add(Slider::new(&mut settings.field.generation.rock_proba, 0.0..=1.0)
                            .text("Rock Probability"));
                        columns[1].add(Slider::new(&mut settings.field.generation.tree_proba, 0.0..=1.0)
                            .text("Tree Probability"));
                        columns[1].add(Slider::new(&mut settings.field.generation.iron_proba, 0.0..=1.0)
                            .text("Iron Probability"));
                        columns[1].add(Slider::new(&mut settings.field.generation.diamond_proba, 0.0..=1.0)
                            .text("Diamond Probability"));
                        columns[1].add(Slider::new(&mut settings.field.generation.structures.robot_proba, 0.0..=1.0)
                            .text("Robot Probability"));
                        columns[1].checkbox(&mut settings.field.from_test_chunk, "Start with Test Chunk");

                        columns[1].label("Player:");
                        columns[1].checkbox(&mut settings.player.cheating_start, "Cheating Start");
                        columns[1].add(Slider::new(&mut settings.player.arrow_break_chance, 0.0..=1.0)
                            .text("Arrow Break Chance"));

                        // Add a "Back" button at the end of the settings menu
                        columns[1].horizontal(|ui| {
                            if ui.button("Back").clicked() {
                                // Close the settings menu
                                self.settings_menu_open = false;
                            }
                        });
                    } else {
                        columns[1].label("A short introduction to the game:");
                        columns[1].label("This is a survival game, that take inspiration from Minecraft and Rogue.");
                        columns[1].label("You will have to craft items, fight enemies and explore the world.");
                        columns[1].label("You will be able to build your own base and defend it from enemies.");
                        columns[1].label("Your attack damage depends on the items in your inventory.");
                        columns[1].label("Better pickaxes will allow you to mine more types of blocks.");
                        columns[1].label("Good luck!");
                    }
                });
            });
    }

    fn render_difficulty_buttons(ui: &mut egui::Ui, settings: &mut Settings) {
        ui.label("Difficulty:");
        ui.horizontal(|ui| {
            if ui.button(RichText::new("Easy").color(Color32::GREEN)).clicked() {
                settings.player.cheating_start = true;
                settings.player.arrow_break_chance = 0.1;
                settings.mobs.spawning.initial_hostile_per_chunk = 0.1;
                settings.mobs.spawning.base_night_amount = 3;
                settings.mobs.spawning.max_mobs_on_chunk = 2;
            }
            if ui.button(RichText::new("Medium").color(Color32::YELLOW)).clicked() {
                settings.player.cheating_start = false;
                settings.player.arrow_break_chance = 0.3;
                settings.mobs.spawning.initial_hostile_per_chunk = 0.2;
                settings.mobs.spawning.base_night_amount = 5;
                settings.mobs.spawning.max_mobs_on_chunk = 3;
            }
            if ui.button(RichText::new("Hard").color(Color32::RED)).clicked() {
                settings.player.cheating_start = false;
                settings.player.arrow_break_chance = 0.5;
                settings.mobs.spawning.initial_hostile_per_chunk = 0.3;
                settings.mobs.spawning.base_night_amount = 7;
                settings.mobs.spawning.max_mobs_on_chunk = 4;
            }
        });
    }

}
