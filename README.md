# Top-down view Minecraft

This is my clone of Minecraft, built without a game engine, purely on WGPU (and some egui for interface).

It is a turn-based, 2.5d game: the field looks 2d, but you can actually break the top block, or place a block on top of an existing one.

The grey circle indicates the height of the top block (the one you see). The darker the circle, the lower the block.
<img width="51" alt="image" src="https://user-images.githubusercontent.com/27450370/198854409-9959c002-301e-4873-b9f6-7f2131c82fad.png">

The game has crafting, and for some items you will need a crafting table nearby.

### AI-friendly
Part of the purpose of this project is to build a training environment for ML agents.

Such evironment would be similar to the famous [MineRL](https://minerl.io/), as it could have the same complexity in terms of progression: 
mine tree -> make crafting table -> make pickaxe -> etc. 

But would require no computer vision, and would run significantly faster than the original.

## Gameplay screenshots:
<img width="798" alt="image" src="https://user-images.githubusercontent.com/27450370/198854311-ff9dbd08-57dd-4f9c-a50f-a4f56790eb53.png">

<img width="801" alt="image" src="https://user-images.githubusercontent.com/27450370/198854546-10c64335-58cb-4a22-93f7-06bdbebad75b.png">

![telegram-cloud-photo-size-2-5316522196908228153-y](https://user-images.githubusercontent.com/27450370/198854616-e86cb3e1-629e-493f-a5ca-fa2be66e9466.jpg)

