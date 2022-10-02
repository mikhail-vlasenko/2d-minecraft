use wgpu::{BindGroup, BindGroupLayout, Device};
use crate::graphics::state::State;
use crate::graphics::texture::Texture;
use crate::Material;


/// Creates and stores wgpu texture bind groups.
pub struct TextureBindGroups {
    pub grass: BindGroup,
    pub stone: BindGroup,
    pub tree_log: BindGroup,
    pub bedrock: BindGroup,
    pub planks: BindGroup,
    pub player: BindGroup,
    pub depth_indicators: [BindGroup; 4],
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
            &device, &queue, include_bytes!("../../res/mc_grass.png"), "mc_grass.png",
        ).unwrap();
        let grass = Self::make_bind_group(
            "grass_bind_group", &grass_texture, &device, &bind_group_layout,
        );

        let stone_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_stone.png"), "mc_stone.png",
        ).unwrap();
        let stone = Self::make_bind_group(
            "stone_bind_group", &stone_texture, &device, &bind_group_layout,
        );

        let tree_log_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_tree_log.png"), "mc_tree_log.png",
        ).unwrap();
        let tree_log = Self::make_bind_group(
            "tree_log_bind_group", &tree_log_texture, &device, &bind_group_layout,
        );

        let bedrock_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_bedrock.png"), "bedrock.png",
        ).unwrap();
        let bedrock = Self::make_bind_group(
            "bedrock_bind_group", &bedrock_texture, &device, &bind_group_layout,
        );

        let planks_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/mc_planks.png"), "planks.png",
        ).unwrap();
        let planks = Self::make_bind_group(
            "planks_bind_group", &planks_texture, &device, &bind_group_layout,
        );

        let player_texture = Texture::from_bytes(
            &device, &queue, include_bytes!("../../res/player_top_view.png"), "player.png",
        ).unwrap();
        let player = Self::make_bind_group(
            "player_bind_group", &player_texture, &device, &bind_group_layout,
        );

        let depth_indicators = Self::init_depth_groups(device, queue, &bind_group_layout);

        TextureBindGroups {
            grass,
            stone,
            tree_log,
            bedrock,
            planks,
            player,
            depth_indicators,
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

    pub fn get_bind_group_for(&self, material: Material) -> &BindGroup {
        use Material::*;
        match material {
            Dirt => &self.grass,
            Stone => &self.stone,
            TreeLog => &self.tree_log,
            Bedrock => &self.bedrock,
            Plank => &self.planks,
        }
    }
}
