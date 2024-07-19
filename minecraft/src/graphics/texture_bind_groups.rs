use wgpu::{BindGroup, BindGroupLayout, Device};
use game_logic::auxiliary::animations::{ProjectileType, TileAnimationType};
use game_logic::crafting::interactable::InteractableKind;
use game_logic::crafting::material::Material;
use game_logic::crafting::texture_material::TextureMaterial;
use game_logic::map_generation::mobs::mob_kind::MobKind;
use crate::graphics::texture::Texture;


/// Creates and stores wgpu texture bind groups.
pub struct TextureBindGroups {
    grass: BindGroup,
    stone: BindGroup,
    tree_log: BindGroup,
    bedrock: BindGroup,
    planks: BindGroup,
    iron_ore: BindGroup,
    crafting_table: BindGroup,
    diamond: BindGroup,
    player: BindGroup,
    depth_indicators: [BindGroup; 4],
    zombie: BindGroup,
    zergling: BindGroup,
    baneling: BindGroup,
    gelatinous_cube: BindGroup,
    cow: BindGroup,
    night: BindGroup,
    red_moon: BindGroup,
    loot_sack: BindGroup,
    arrow: BindGroup,
    crossbow_turret: BindGroup,
    red_hp_bar: BindGroup,
    green_hp_bar: BindGroup,
    yellow_hit: BindGroup,
    red_hit: BindGroup,
    channeling: BindGroup,
    vertical_arrow: BindGroup,
    texture_materials: TextureMaterials,
    pub bind_group_layout: BindGroupLayout,
}

impl TextureBindGroups {
    fn make_layout(device: &Device) -> BindGroupLayout {
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
        })
    }
    fn make_bind_group(
        label: &str, texture: &Texture, device: &Device, texture_bind_group_layout: &BindGroupLayout,
    ) -> BindGroup {
        device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture.sampler),
                    }
                ],
                label: Some(label),
            }
        )
    }

    pub fn new(device: &Device, queue: &wgpu::Queue) -> Self {
        let bind_group_layout = Self::make_layout(device);

        macro_rules! make_bind_group_from_texture {
            ($path:expr) => {{
                let texture = Texture::from_bytes(&device, &queue, include_bytes!($path), "texture.png").unwrap();
                Self::make_bind_group("a_bind_group", &texture, &device, &bind_group_layout)
            }};
        }

        let grass = make_bind_group_from_texture!("../../res/tiles/mc_grass.png");
        let stone = make_bind_group_from_texture!("../../res/tiles/mc_stone.png");
        let tree_log = make_bind_group_from_texture!("../../res/tiles/mc_tree_log.png");
        let bedrock = make_bind_group_from_texture!("../../res/tiles/mc_bedrock.png");
        let planks = make_bind_group_from_texture!("../../res/tiles/mc_planks.png");
        let iron_ore = make_bind_group_from_texture!("../../res/tiles/mc_iron_ore.png");
        let crafting_table = make_bind_group_from_texture!("../../res/tiles/mc_crafting_table.png");
        let diamond = make_bind_group_from_texture!("../../res/tiles/mc_diamond.png");
        let zombie = make_bind_group_from_texture!("../../res/mobs/zombie.png");
        let zergling = make_bind_group_from_texture!("../../res/mobs/zergling.png");
        let baneling = make_bind_group_from_texture!("../../res/mobs/baneling.png");
        let gelatinous_cube = make_bind_group_from_texture!("../../res/mobs/gelatinous_cube.png");
        let cow = make_bind_group_from_texture!("../../res/mobs/mc_cow.png");
        let night = make_bind_group_from_texture!("../../res/transparent gradient.png");
        let red_moon = make_bind_group_from_texture!("../../res/red_moon_grad.png");
        let loot_sack = make_bind_group_from_texture!("../../res/loot sack.png");
        let arrow = make_bind_group_from_texture!("../../res/fat_arrow.png");
        let crossbow_turret = make_bind_group_from_texture!("../../res/interactables/fat_string_crossbow.png");
        let red_hp_bar = make_bind_group_from_texture!("../../res/red_hp_bar.png");
        let green_hp_bar = make_bind_group_from_texture!("../../res/green_hp_bar.png");
        let player = make_bind_group_from_texture!("../../res/player_top_view.png");
        
        let yellow_hit = make_bind_group_from_texture!("../../res/animations/unrolled_yellow_hit.png");
        let red_hit = make_bind_group_from_texture!("../../res/animations/unrolled_red_hit_center_crop.png");
        let channeling = make_bind_group_from_texture!("../../res/animations/blue_channeling.png");
        let vertical_arrow = make_bind_group_from_texture!("../../res/animations/fat_arrow_vertical.png");

        let depth_indicators = Self::init_depth_groups(device, queue, &bind_group_layout);

        let texture_materials = TextureMaterials::new(device, queue, &bind_group_layout);

        TextureBindGroups {
            grass,
            stone,
            tree_log,
            bedrock,
            planks,
            iron_ore,
            crafting_table,
            diamond,
            player,
            depth_indicators,
            zombie,
            zergling,
            baneling,
            gelatinous_cube,
            cow,
            night,
            red_moon,
            loot_sack,
            arrow,
            crossbow_turret,
            red_hp_bar,
            green_hp_bar,
            yellow_hit,
            red_hit,
            channeling,
            vertical_arrow,
            texture_materials,
            bind_group_layout,
        }
    }

    fn init_depth_groups(
        device: &Device, queue: &wgpu::Queue, texture_bind_group_layout: &BindGroupLayout,
    ) -> [BindGroup; 4] {
        macro_rules! make_bind_group_from_texture {
            ($path:expr, $layout:expr) => {{
                let texture = Texture::from_bytes(&device, &queue, include_bytes!($path), "depth.png").unwrap();
                Self::make_bind_group("depth_bind_group", &texture, &device, $layout)
            }};
        }

        let depth1 = make_bind_group_from_texture!("../../res/depth_indicators/depth1.png", &texture_bind_group_layout);
        let depth2 = make_bind_group_from_texture!("../../res/depth_indicators/depth2.png", &texture_bind_group_layout);
        let depth3 = make_bind_group_from_texture!("../../res/depth_indicators/depth3.png", &texture_bind_group_layout);
        let depth4 = make_bind_group_from_texture!("../../res/depth_indicators/depth4.png", &texture_bind_group_layout);

        [depth1, depth2, depth3, depth4]
    }

    pub fn get_bind_group_material(&self, material: Material) -> &BindGroup {
        use Material::*;
        match material {
            Dirt => &self.grass,
            Stone => &self.stone,
            TreeLog => &self.tree_log,
            Bedrock => &self.bedrock,
            Plank => &self.planks,
            IronOre => &self.iron_ore,
            CraftTable => &self.crafting_table,
            Diamond => &self.diamond,
            Texture(t) => self.texture_materials.get_bind_group(t),
        }
    }

    pub fn get_bind_group_mob(&self, mob: MobKind) -> &BindGroup {
        use MobKind::*;
        match mob {
            Zombie => &self.zombie,
            Zergling => &self.zergling,
            Baneling => &self.baneling,
            GelatinousCube => &self.gelatinous_cube,
            Cow => &self.cow,
        }
    }

    pub fn get_bind_group_interactable(&self, interactable: InteractableKind) -> &BindGroup {
        use InteractableKind::*;
        match interactable {
            CrossbowTurret => &self.crossbow_turret,
        }
    }

    pub fn get_bind_group_depth(&self, depth: usize) -> &BindGroup {
        match depth {
            0 => &self.depth_indicators[0],
            1 => &self.depth_indicators[1],
            2 => &self.depth_indicators[2],
            3 => &self.depth_indicators[3],
            _ => panic!("Depth {} is not supported", depth)
        }
    }

    pub fn get_bind_group_player(&self) -> &BindGroup {
        &self.player
    }

    pub fn get_bind_group_night(&self) -> &BindGroup {
        &self.night
    }

    pub fn get_bind_group_red_moon(&self) -> &BindGroup {
        &self.red_moon
    }

    pub fn get_bind_group_loot(&self) -> &BindGroup {
        &self.loot_sack
    }

    pub fn get_bind_group_arrow(&self) -> &BindGroup {
        &self.arrow
    }

    pub fn get_bind_group_hp_bar(&self, red: bool) -> &BindGroup {
        if red {
            &self.red_hp_bar
        } else {
            &self.green_hp_bar
        }
    }
    
    pub fn get_bind_group_animation(&self, tile_animation_type: TileAnimationType) -> &BindGroup {
        use TileAnimationType::*;
        match tile_animation_type {
            YellowHit => &self.yellow_hit,
            RedHit => &self.red_hit,
            Channelling => &self.channeling,
        }
    }
    
    pub fn get_bind_group_projectile(&self, projectile_type: ProjectileType) -> &BindGroup {
        use ProjectileType::*;
        match projectile_type {
            Arrow => &self.vertical_arrow,
            GelatinousCube => &self.gelatinous_cube,
        }
    }
}

struct TextureMaterials {
    unknown: BindGroup,
    robot_tr: BindGroup,
    robot_tl: BindGroup,
    robot_br: BindGroup,
    robot_bl: BindGroup,
}

impl TextureMaterials {
    pub fn new(device: &Device, queue: &wgpu::Queue, bind_group_layout: &BindGroupLayout) -> Self {
        macro_rules! make_bind_group_from_texture {
            ($path:expr) => {{
                let texture = Texture::from_bytes(&device, &queue, include_bytes!($path), "texture.png").unwrap();
                TextureBindGroups::make_bind_group("texture_material_bind_group", &texture, &device, &bind_group_layout)
            }};
        }

        let unknown = make_bind_group_from_texture!("../../res/tiles/texture_materials/unknown.png");
        let robot_tr = make_bind_group_from_texture!("../../res/tiles/texture_materials/right_top_war_robot.png");
        let robot_tl = make_bind_group_from_texture!("../../res/tiles/texture_materials/left_top_war_robot.png");
        let robot_br = make_bind_group_from_texture!("../../res/tiles/texture_materials/right_bot_war_robot.png");
        let robot_bl = make_bind_group_from_texture!("../../res/tiles/texture_materials/left_bot_war_robot.png");

        TextureMaterials {
            unknown,
            robot_tr,
            robot_tl,
            robot_br,
            robot_bl,
        }
    }

    pub fn get_bind_group(&self, texture_material: TextureMaterial) -> &BindGroup {
        use TextureMaterial::*;
        match texture_material {
            Unknown => &self.unknown,
            RobotTR => &self.robot_tr,
            RobotTL => &self.robot_tl,
            RobotBR => &self.robot_br,
            RobotBL => &self.robot_bl,
        }
    }
}
