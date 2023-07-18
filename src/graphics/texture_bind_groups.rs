use wgpu::{BindGroup, BindGroupLayout, Device};
use crate::crafting::interactable::InteractableKind;
use crate::crafting::material::Material;
use crate::crafting::texture_material::TextureMaterial;
use crate::graphics::texture::Texture;
use crate::map_generation::mobs::mob_kind::MobKind;


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
    cow: BindGroup,
    night: BindGroup,
    red_moon: BindGroup,
    loot_sack: BindGroup,
    arrow: BindGroup,
    crossbow_turret: BindGroup,
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

        let grass_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_grass.png"), "mc_grass.png",
        ).unwrap();
        let grass = Self::make_bind_group(
            "grass_bind_group", &grass_texture, &device, &bind_group_layout,
        );

        let stone_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_stone.png"), "mc_stone.png",
        ).unwrap();
        let stone = Self::make_bind_group(
            "stone_bind_group", &stone_texture, &device, &bind_group_layout,
        );

        let tree_log_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_tree_log.png"), "mc_tree_log.png",
        ).unwrap();
        let tree_log = Self::make_bind_group(
            "tree_log_bind_group", &tree_log_texture, &device, &bind_group_layout,
        );

        let bedrock_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_bedrock.png"), "bedrock.png",
        ).unwrap();
        let bedrock = Self::make_bind_group(
            "bedrock_bind_group", &bedrock_texture, &device, &bind_group_layout,
        );

        let planks_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_planks.png"), "planks.png",
        ).unwrap();
        let planks = Self::make_bind_group(
            "planks_bind_group", &planks_texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_iron_ore.png"), "texture.png",
        ).unwrap();
        let iron_ore = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_crafting_table.png"), "texture.png",
        ).unwrap();
        let crafting_table = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/mc_diamond.png"), "texture.png",
        ).unwrap();
        let diamond = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mobs/zombie.png"), "texture.png",
        ).unwrap();
        let zombie = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mobs/zergling.png"), "texture.png",
        ).unwrap();
        let zergling = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mobs/baneling.png"), "texture.png",
        ).unwrap();
        let baneling = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mobs/mc_cow.png"), "texture.png",
        ).unwrap();
        let cow = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/transparent gradient.png"), "texture.png",
        ).unwrap();
        let night = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/red_moon_grad.png"), "texture.png",
        ).unwrap();
        let red_moon = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/loot sack.png"), "texture.png",
        ).unwrap();
        let loot_sack = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/fat_arrow.png"), "texture.png",
        ).unwrap();
        let arrow = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/interactables/fat_string_crossbow.png"), "texture.png",
        ).unwrap();
        let crossbow_turret = Self::make_bind_group(
            "a_bind_group", &texture, &device, &bind_group_layout,
        );

        let player_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/player_top_view.png"), "player.png",
        ).unwrap();
        let player = Self::make_bind_group(
            "player_bind_group", &player_texture, &device, &bind_group_layout,
        );

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
            cow,
            night,
            red_moon,
            loot_sack,
            arrow,
            crossbow_turret,
            texture_materials,
            bind_group_layout,
        }
    }

    fn init_depth_groups(
        device: &Device, queue: &wgpu::Queue, texture_bind_group_layout: &BindGroupLayout,
    ) -> [BindGroup; 4] {
        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/depth_indicators/depth1.png"), "depth.png",
        ).unwrap();
        let depth1 = Self::make_bind_group(
            "depth_bind_group", &texture, &device, &texture_bind_group_layout,
        );
        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/depth_indicators/depth2.png"), "depth.png",
        ).unwrap();
        let depth2 = Self::make_bind_group(
            "depth_bind_group", &texture, &device, &texture_bind_group_layout,
        );
        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/depth_indicators/depth3.png"), "depth.png",
        ).unwrap();
        let depth3 = Self::make_bind_group(
            "depth_bind_group", &texture, &device, &texture_bind_group_layout,
        );
        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/depth_indicators/depth4.png"), "depth.png",
        ).unwrap();
        let depth4 = Self::make_bind_group(
            "depth_bind_group", &texture, &device, &texture_bind_group_layout,
        );

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
        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/texture_materials/unknown.png"), "texture.png",
        ).unwrap();
        let unknown = TextureBindGroups::make_bind_group(
            "a_bind_group", &texture, &device, bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/texture_materials/right_top_war_robot.png"), "texture.png",
        ).unwrap();
        let robot_tr = TextureBindGroups::make_bind_group(
            "a_bind_group", &texture, &device, bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/texture_materials/left_top_war_robot.png"), "texture.png",
        ).unwrap();
        let robot_tl = TextureBindGroups::make_bind_group(
            "a_bind_group", &texture, &device, bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/texture_materials/right_bot_war_robot.png"), "texture.png",
        ).unwrap();
        let robot_br = TextureBindGroups::make_bind_group(
            "a_bind_group", &texture, &device, bind_group_layout,
        );

        let texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/tiles/texture_materials/left_bot_war_robot.png"), "texture.png",
        ).unwrap();
        let robot_bl = TextureBindGroups::make_bind_group(
            "a_bind_group", &texture, &device, bind_group_layout,
        );

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
